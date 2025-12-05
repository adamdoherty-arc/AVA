"""
AVA Trading Platform - Main Application
========================================

Modern FastAPI application with:
- Async database connections (asyncpg)
- Distributed caching (Redis)
- Observability (OpenTelemetry + Prometheus)
- Structured logging (structlog)
- Circuit breaker pattern
- Rate limiting
- Comprehensive error handling
- Background task management

Author: AVA Trading Platform
Updated: 2025-11-29
"""

from contextlib import asynccontextmanager
import asyncio
from datetime import datetime, time as dt_time, timedelta
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
import uvicorn

# Modern infrastructure
from backend.infrastructure.cache import init_cache
from backend.infrastructure.database import init_database, get_database
from backend.infrastructure.observability import (
    setup_observability,
    get_logger,
    metrics,
    health_checker,
    correlation_context,
    generate_correlation_id,
)
from backend.infrastructure.errors import (
    register_exception_handlers,
    AVAError,
    ErrorCode,
)
from backend.config import settings as app_settings

# Routers
from backend.routers import (
    sports, predictions, dashboard, chat, research, portfolio, strategy,
    scanner, agents, earnings, xtrades, system, technicals, knowledge,
    enhancements, analytics, options, subscriptions, cache, discord, settings,
    integration_test, watchlist, qa_dashboard, goals, briefings, automations,
    # Advanced Technical Analysis routers
    smart_money, advanced_technicals, options_indicators,
    # Sports Streaming (SSE)
    sports_streaming,
    # Sports V2 - Modern async implementation
    sports_v2,
    # Notifications (Telegram)
    notifications,
    # Portfolio V2 - Modern optimized infrastructure
    portfolio_v2,
    # Master Orchestrator - Feature specs and codebase intelligence
    orchestrator,
    # Stocks Tile Hub - AI-scored stock tiles
    stocks_tiles,
    # DeepSeek R1 32B Deep Reasoning API
    reasoning,
    # Stock Universe - Complete stock/ETF database
    universe
)

# Initialize observability early
setup_observability()
logger = get_logger(__name__)

# Background task references for cleanup
_background_tasks = []
_task_restart_counts: dict = {}  # Track restart counts per task
MAX_TASK_RESTARTS = 5  # Maximum restarts before giving up


async def resilient_task_wrapper(task_name: str, task_func, *args, **kwargs):
    """
    Wrapper that makes background tasks resilient with auto-restart.

    Features:
    - Automatic restart on failure with exponential backoff
    - Maximum restart limit to prevent infinite loops
    - Detailed logging of failures and restarts
    """
    global _task_restart_counts
    _task_restart_counts[task_name] = 0
    backoff_seconds = 60  # Start with 1 minute

    while _task_restart_counts[task_name] < MAX_TASK_RESTARTS:
        try:
            await task_func(*args, **kwargs)
            # If task completes normally (shouldn't happen for long-running tasks)
            break
        except asyncio.CancelledError:
            logger.info(f"resilient_task_cancelled", task=task_name)
            break
        except Exception as e:
            _task_restart_counts[task_name] += 1
            logger.error(
                "resilient_task_failed",
                task=task_name,
                error=str(e),
                restart_count=_task_restart_counts[task_name],
                max_restarts=MAX_TASK_RESTARTS
            )

            if _task_restart_counts[task_name] >= MAX_TASK_RESTARTS:
                logger.critical(
                    "resilient_task_max_restarts_reached",
                    task=task_name,
                    message="Task will not be restarted"
                )
                break

            # Exponential backoff with cap at 30 minutes
            wait_time = min(backoff_seconds * (2 ** (_task_restart_counts[task_name] - 1)), 1800)
            logger.info(
                "resilient_task_restarting",
                task=task_name,
                wait_seconds=wait_time
            )

            # Use cancellable sleep to allow clean shutdown
            try:
                await asyncio.sleep(wait_time)
            except asyncio.CancelledError:
                logger.info("resilient_task_sleep_cancelled", task=task_name)
                raise  # Re-raise to exit the wrapper properly


async def periodic_cache_cleanup():
    """
    Background task to periodically clean up unused cache locks.
    Runs every 5 minutes to prevent memory leaks.
    """
    from backend.infrastructure.cache import get_cache

    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            cache = get_cache()
            if hasattr(cache, '_lock_manager'):
                removed = await cache._lock_manager.cleanup_unused()
                if removed > 0:
                    logger.info("cache_locks_cleaned", count=removed)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("cache_cleanup_error", error=str(e))


async def periodic_rag_sync():
    """
    Background task to sync RAG knowledge base.
    Runs every hour to keep knowledge base updated with:
    - Earnings transcripts
    - Discord premium signals
    - XTrades trader messages
    - News articles
    """
    # Wait 2 minutes before first sync to let other services initialize
    await asyncio.sleep(120)

    while True:
        try:
            logger.info("rag_sync_starting")

            # Run sync in thread pool since RAG service uses blocking I/O
            def sync_rag():
                from src.rag.rag_service import get_rag_service
                rag = get_rag_service()
                return rag.sync_from_database("all")

            stats = await asyncio.get_event_loop().run_in_executor(None, sync_rag)

            total = (stats['earnings_synced'] + stats['discord_synced'] +
                    stats['xtrades_synced'] + stats['news_synced'])

            logger.info(
                "rag_sync_complete",
                total_synced=total,
                earnings=stats['earnings_synced'],
                discord=stats['discord_synced'],
                xtrades=stats['xtrades_synced'],
                news=stats['news_synced'],
                errors=len(stats.get('errors', []))
            )

            # Run every hour
            await asyncio.sleep(3600)

        except asyncio.CancelledError:
            logger.info("rag_sync_cancelled")
            break
        except Exception as e:
            logger.error("rag_sync_error", error=str(e))
            # Wait 15 minutes before retry on error
            await asyncio.sleep(900)


async def auto_sync_earnings():
    """
    Background task to auto-sync earnings data.

    Runs:
    - On startup (initial sync)
    - Daily at 6:00 AM ET (before market open)

    This ensures earnings data is always fresh when users access the platform.
    """
    from src.earnings_manager import EarningsManager

    # Initial sync on startup
    try:
        logger.info("Auto-sync: Running initial earnings sync...")
        manager = EarningsManager()
        # Run sync in thread pool since EarningsManager is synchronous
        await asyncio.to_thread(manager.sync_finnhub_earnings)
        logger.info("Auto-sync: Initial earnings sync complete")
    except Exception as e:
        logger.error(f"Auto-sync: Initial earnings sync failed: {e}")

    # Schedule daily sync
    while True:
        try:
            now = datetime.now()
            # Calculate time until 6 AM tomorrow
            target = datetime.combine(
                now.date(),
                dt_time(hour=6, minute=0)
            )
            if now.hour >= 6:
                # Already past 6 AM, schedule for tomorrow
                target = target.replace(day=now.day + 1)

            wait_seconds = (target - now).total_seconds()
            logger.info(
                f"Auto-sync: Next earnings sync in {wait_seconds/3600:.1f} hours"
            )
            await asyncio.sleep(wait_seconds)

            # Run sync
            logger.info("Auto-sync: Running scheduled earnings sync...")
            manager = EarningsManager()
            await asyncio.to_thread(manager.sync_finnhub_earnings)
            logger.info("Auto-sync: Scheduled earnings sync complete")

        except asyncio.CancelledError:
            logger.info("Auto-sync: Earnings sync task cancelled")
            break
        except Exception as e:
            logger.error(f"Auto-sync: Earnings sync error: {e}")
            # Wait 1 hour before retrying on error
            await asyncio.sleep(3600)


async def auto_sync_espn_games():
    """
    Background task to auto-sync ESPN game data to database.

    Runs:
    - Immediately on startup (initial sync)
    - Every 10 minutes during active hours (8 AM - 12 AM)

    ESPN API is FREE - no API key required!
    This populates the sports game tables that the frontend tiles read from.
    """
    from backend.routers.sports import sync_sports_data

    # Initial sync on startup
    try:
        logger.info("Auto-sync: Running initial ESPN games sync...")
        result = await sync_sports_data(sport="ALL")
        logger.info(f"Auto-sync: ESPN initial sync complete - {result.get('total_synced', 0)} games synced")
    except Exception as e:
        logger.error(f"Auto-sync: Initial ESPN sync failed: {e}")

    # Periodic sync
    while True:
        try:
            now = datetime.now()
            # Sync during active hours (8 AM - 12 AM)
            if 8 <= now.hour <= 23:
                await asyncio.sleep(600)  # Wait 10 minutes first
                logger.info("Auto-sync: Running ESPN games sync...")
                result = await sync_sports_data(sport="ALL")
                logger.info(f"Auto-sync: ESPN sync complete - {result.get('total_synced', 0)} games synced")
            else:
                # Outside active hours, check every 30 minutes
                await asyncio.sleep(1800)

        except asyncio.CancelledError:
            logger.info("Auto-sync: ESPN games sync task cancelled")
            break
        except Exception as e:
            logger.error(f"Auto-sync: ESPN games sync error: {e}")
            await asyncio.sleep(300)


async def auto_sync_sports_odds():
    """
    Background task to auto-sync sports odds data from Kalshi.

    Runs every 15 minutes during active hours (9 AM - 11 PM).
    This keeps odds data fresh for the sports betting features.
    Uses Kalshi public API (no API key required) instead of The Odds API.

    IMPORTANT: Runs with timeout to prevent blocking other requests.
    """
    from backend.routers.sports import sync_kalshi_odds
    import concurrent.futures

    # Wait for ESPN sync to populate games first
    await asyncio.sleep(30)

    while True:
        try:
            now = datetime.now()
            # Only sync during active hours (9 AM - 11 PM)
            if 9 <= now.hour <= 23:
                logger.info("Auto-sync: Running Kalshi odds sync...")
                try:
                    # Run with 30 second timeout to prevent blocking
                    result = await asyncio.wait_for(sync_kalshi_odds(), timeout=30.0)
                    logger.info(f"Auto-sync: Kalshi sync complete - {result.get('synced', 0)} odds synced")
                except asyncio.TimeoutError:
                    logger.warning("Auto-sync: Kalshi sync timed out after 30 seconds")

            # Wait 15 minutes
            await asyncio.sleep(900)

        except asyncio.CancelledError:
            logger.info("Auto-sync: Sports odds sync task cancelled")
            break
        except Exception as e:
            logger.error(f"Auto-sync: Sports odds sync error: {e}")
            # Wait 5 minutes before retrying on error
            await asyncio.sleep(300)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events for:
    - Database connection pool initialization
    - Redis cache initialization
    - Background sync tasks (earnings, sports odds)
    - Metrics initialization
    """
    global _background_tasks

    # ==========================================================================
    # STARTUP
    # ==========================================================================
    logger.info(
        "ava_platform_starting",
        version=app_settings.APP_NAME,
        environment="development" if app_settings.DEBUG else "production",
    )

    # Initialize async database connection pool
    try:
        db = await init_database()
        db_stats = db.get_stats()
        logger.info(
            "database_connected",
            pool_size=db_stats["pool_stats"]["size"],
            max_size=db_stats["pool_stats"]["max_size"],
        )
        # Warmup connection pool to prevent cold-start latency
        warmed = await db.warmup_pool()
        logger.info("database_pool_warmed", connections=warmed)
    except Exception as e:
        logger.error("database_initialization_failed", error=str(e))
        # Continue - some features may work without DB

    # Initialize Redis cache connection
    try:
        cache = await init_cache()
        cache_stats = cache.get_stats()
        if cache_stats["connected"]:
            logger.info(
                "cache_connected",
                backend=cache_stats["backend"],
            )
        else:
            logger.warning("cache_fallback_active", backend="in-memory")
    except Exception as e:
        logger.error("cache_initialization_failed", error=str(e))

    # Start background sync tasks with resilient wrappers
    logger.info("background_tasks_starting")
    global _background_tasks
    _background_tasks = [
        asyncio.create_task(
            resilient_task_wrapper("earnings_sync", auto_sync_earnings),
            name="earnings_sync"
        ),
        asyncio.create_task(
            resilient_task_wrapper("espn_games_sync", auto_sync_espn_games),
            name="espn_games_sync"
        ),
        asyncio.create_task(
            resilient_task_wrapper("sports_odds_sync", auto_sync_sports_odds),
            name="sports_odds_sync"
        ),
        asyncio.create_task(periodic_cache_cleanup(), name="cache_cleanup"),
        asyncio.create_task(
            resilient_task_wrapper("rag_knowledge_sync", periodic_rag_sync),
            name="rag_knowledge_sync"
        ),
    ]
    logger.info("background_tasks_started", count=len(_background_tasks))

    # Start positions sync service (30-minute Robinhood sync for non-blocking UI)
    try:
        from backend.services.positions_sync_service import start_positions_sync_service
        await start_positions_sync_service()
        logger.info("positions_sync_service_started")
    except Exception as e:
        logger.error("positions_sync_service_failed", error=str(e))
        # Continue - positions will fall back to live API

    logger.info("ava_platform_ready")

    yield  # Application runs here

    # ==========================================================================
    # SHUTDOWN
    # ==========================================================================
    logger.info("ava_platform_shutting_down")

    # Stop positions sync service
    try:
        from backend.services.positions_sync_service import stop_positions_sync_service
        await stop_positions_sync_service()
        logger.info("positions_sync_service_stopped")
    except Exception as e:
        logger.warning("positions_sync_service_stop_error", error=str(e))

    # Cancel background tasks gracefully
    for task in _background_tasks:
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    # Close database connections
    try:
        db = await get_database()
        await db.disconnect()
        logger.info("database_disconnected")
    except Exception as e:
        logger.warning("database_disconnect_error", error=str(e))

    logger.info("ava_platform_shutdown_complete")


app = FastAPI(
    title="AVA Trading Platform API",
    description="Backend API for AVA - AI-Powered Trading Platform with Multi-Agent Intelligence",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# =============================================================================
# MIDDLEWARE STACK (order matters - first added = last executed)
# =============================================================================

# GZip compression for responses > 1KB
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS Configuration - Uses centralized config from backend/config.py
# Origins are defined in app_settings.CORS_ORIGINS and parsed by cors_origins_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=app_settings.cors_origins_list,
    allow_credentials=app_settings.CORS_ALLOW_CREDENTIALS,
    # Security: Use explicit methods from config instead of wildcard
    allow_methods=app_settings.cors_methods_list,
    # Security: Use explicit headers from config instead of wildcard
    allow_headers=app_settings.cors_headers_list,
    expose_headers=["X-Correlation-ID", "X-Request-Duration"],
)


# Global request timeout middleware
REQUEST_TIMEOUT_SECONDS = 60  # 60 second timeout for all requests


@app.middleware("http")
async def request_timeout_middleware(request: Request, call_next):
    """
    Enforce global request timeout to prevent hung requests.

    Exempt paths:
    - WebSocket upgrades
    - Streaming endpoints
    - Health checks
    """
    # Exempt paths from timeout
    exempt_paths = ["/api/health", "/ws", "/api/chat/stream", "/api/sports/stream"]
    if any(request.url.path.startswith(p) for p in exempt_paths):
        return await call_next(request)

    try:
        return await asyncio.wait_for(
            call_next(request),
            timeout=REQUEST_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        logger.error(
            "request_timeout",
            path=request.url.path,
            method=request.method,
            timeout_seconds=REQUEST_TIMEOUT_SECONDS
        )
        return JSONResponse(
            status_code=504,
            content={
                "error": "Request timeout",
                "detail": f"Request exceeded {REQUEST_TIMEOUT_SECONDS} second timeout",
                "path": request.url.path
            }
        )


# Request logging and correlation middleware
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Add correlation ID, logging, and timing to all requests."""
    import time

    # Generate or extract correlation ID
    correlation_id = request.headers.get("X-Correlation-ID") or generate_correlation_id()

    # Start timing
    start_time = time.perf_counter()

    # Process request with correlation context
    with correlation_context(correlation_id):
        logger.info(
            "request_started",
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )

        try:
            response = await call_next(request)

            # Calculate duration
            duration = time.perf_counter() - start_time
            duration_ms = round(duration * 1000, 2)

            # Add headers
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Request-Duration"] = f"{duration_ms}ms"

            # Log completion
            logger.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=duration_ms,
            )

            # Record metrics
            metrics.record_request(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code,
                duration=duration,
            )

            return response

        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error(
                "request_failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
                duration_ms=round(duration * 1000, 2),
            )
            raise


# Register exception handlers
register_exception_handlers(app)

# Core routers
app.include_router(sports.router)
app.include_router(sports_v2.router)  # Sports V2 - Modern async implementation
app.include_router(sports_streaming.router)  # SSE streaming for AI predictions
app.include_router(predictions.router)
app.include_router(dashboard.router)
app.include_router(chat.router)
app.include_router(research.router)
app.include_router(portfolio.router)
# app.include_router(portfolio.positions_router)  # Alias for /api/positions - TODO: add to portfolio.py
app.include_router(strategy.router)
# app.include_router(strategy.strategies_router)  # strategies endpoint - TODO: add to strategy.py
app.include_router(scanner.router)
app.include_router(agents.router)
app.include_router(earnings.router)
app.include_router(xtrades.router)

# New routers
app.include_router(system.router)
app.include_router(technicals.router)
app.include_router(knowledge.router)
app.include_router(enhancements.router)
app.include_router(analytics.router)
app.include_router(options.router)
app.include_router(subscriptions.router)
app.include_router(cache.router)
app.include_router(discord.router)
app.include_router(settings.router)
app.include_router(integration_test.router)
app.include_router(watchlist.router)

# Advanced Technical Analysis routers
app.include_router(smart_money.router)
app.include_router(advanced_technicals.router)
app.include_router(options_indicators.router)

# QA Dashboard
app.include_router(qa_dashboard.router)

# Goal tracking
app.include_router(goals.router)

# Briefings and reports
app.include_router(briefings.router)

# Automations Developer Console
app.include_router(automations.router)

# Notifications (Telegram)
app.include_router(notifications.router)

# Portfolio V2 - Modern optimized infrastructure with:
# - Distributed caching (Redis)
# - Circuit breaker pattern
# - Rate limiting
# - Parallel batch fetching
# - WebSocket real-time updates
# - Advanced risk models (Cornish-Fisher VaR, Monte Carlo)
# - Background task management
app.include_router(portfolio_v2.router)

# Master Orchestrator - Feature specs and codebase intelligence
# - Natural language queries about AVA
# - Semantic search across all features
# - Dependency analysis
# - Efficiency gap detection
# - Impact analysis for changes
app.include_router(orchestrator.router)

# Stocks Tile Hub - AI-scored stock tiles
# - Composite AI score (0-100) combining prediction, technicals, volatility
# - Batch scoring for watchlists
# - SSE streaming analysis
# - Custom watchlist management
app.include_router(stocks_tiles.router)

# DeepSeek R1 32B Deep Reasoning API - Chain-of-thought reasoning
app.include_router(reasoning.router)

# Stock Universe - Complete stock/ETF database with 60+ data points per stock
# - Full company information (fundamentals, technicals, valuations)
# - Advanced filtering by sector, industry, market cap, volume
# - Search and symbol lookup
app.include_router(universe.router)


# =============================================================================
# ROOT & HEALTH ENDPOINTS
# =============================================================================


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/api/health")
async def health_check():
    """
    Basic health check endpoint.

    Returns 200 if the service is running.
    Use /api/health/ready for comprehensive readiness check.
    """
    return {
        "status": "healthy",
        "service": "AVA Trading Platform API",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/health/live")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.

    Returns 200 if the service process is alive.
    This should always return quickly - no dependency checks.
    """
    return {"status": "alive"}


@app.get("/api/health/ready")
async def readiness_check():
    """
    Kubernetes readiness probe endpoint.

    Performs comprehensive health checks on all dependencies:
    - Database connectivity
    - Redis connectivity
    - External services

    Returns 200 if ready to handle traffic, 503 otherwise.
    """
    report = await health_checker.run_all()

    status_code = 200 if report.status.value == "healthy" else 503

    return JSONResponse(
        status_code=status_code,
        content=report.to_dict(),
    )


@app.get("/api/metrics")
async def get_metrics(request: Request):
    """
    Get application metrics.

    Returns Prometheus-compatible metrics for monitoring.

    Security: Requires X-Metrics-Token header matching METRICS_TOKEN env var,
    or request from localhost/internal network.
    """
    import os

    # Security check - require token or localhost
    metrics_token = os.getenv("METRICS_TOKEN")
    client_host = request.client.host if request.client else None
    is_localhost = client_host in ("127.0.0.1", "localhost", "::1", None)
    request_token = request.headers.get("X-Metrics-Token")

    if metrics_token and not is_localhost:
        if request_token != metrics_token:
            logger.warning(
                "metrics_access_denied",
                client=client_host,
                reason="invalid_token"
            )
            return JSONResponse(
                status_code=403,
                content={"error": "Access denied. Provide valid X-Metrics-Token header."}
            )

    try:
        # Get database stats
        db = await get_database()
        db_stats = db.get_stats()
    except Exception:
        db_stats = {"error": "Database unavailable"}

    # Get cache stats
    try:
        from backend.infrastructure.cache import get_cache
        cache = get_cache()
        cache_stats = cache.get_stats()
    except Exception:
        cache_stats = {"error": "Cache unavailable"}

    return {
        "database": db_stats,
        "cache": cache_stats,
        "task_restart_counts": _task_restart_counts,
        "background_tasks": [
            {
                "name": task.get_name(),
                "done": task.done(),
                "cancelled": task.cancelled(),
            }
            for task in _background_tasks
        ],
    }


# =============================================================================
# SERVER ENTRY POINT
# =============================================================================


def run_server():
    """Run the server programmatically."""
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=app_settings.SERVER_PORT,  # Uses centralized config
        reload=app_settings.DEBUG,
        log_level="info",
        access_log=True,
    )


def main():
    """Main entry point with singleton enforcement."""
    parser = argparse.ArgumentParser(description='AVA Backend Server')
    parser.add_argument('--force', action='store_true',
                        help='Force start (kill existing instance)')
    parser.add_argument('--status', action='store_true',
                        help='Show status of backend server')
    parser.add_argument('--stop', action='store_true',
                        help='Stop running backend server')
    parser.add_argument('--port', type=int, default=app_settings.SERVER_PORT,
                        help=f'Port to run on (default: {app_settings.SERVER_PORT})')
    parser.add_argument('--no-reload', action='store_true',
                        help='Disable auto-reload')
    parser.add_argument('--json', action='store_true',
                        help='Output status as JSON (for --status)')
    parser.add_argument('--graceful', action='store_true', default=True,
                        help='Use graceful shutdown (default: True)')

    args = parser.parse_args()

    # Import process manager
    try:
        from src.utils.process_manager import ProcessManager, stop_service, print_status
        pm = ProcessManager('backend', port=args.port)
    except ImportError:
        logger.warning("ProcessManager not available, running without singleton enforcement")
        pm = None

    # Handle status command
    if args.status:
        if pm:
            print_status('backend', port=args.port, as_json=args.json)
        else:
            print("ProcessManager not available")
        sys.exit(0)

    # Handle stop command
    if args.stop:
        if pm:
            stop_service('backend', port=args.port, graceful=args.graceful)
        else:
            print("ProcessManager not available")
        sys.exit(0)

    # Enforce singleton
    if pm:
        is_running, existing_pid = pm.is_already_running()

        if is_running and not args.force:
            print(f"\nERROR: Backend server is already running!")
            if existing_pid:
                print(f"  PID: {existing_pid}")
            print(f"  Port: {args.port}")
            print(f"\nOptions:")
            print(f"  1. Stop existing:  python -m backend.main --stop")
            print(f"  2. Force restart:  python -m backend.main --force")
            print(f"  3. Check status:   python -m backend.main --status")
            print(f"  4. JSON status:    python -m backend.main --status --json")
            sys.exit(1)

        if not pm.acquire_lock(force=args.force):
            print("Failed to acquire lock")
            sys.exit(1)

    try:
        print(f"\nStarting AVA Backend Server on port {args.port}...")
        uvicorn.run(
            "backend.main:app",
            host="0.0.0.0",
            port=args.port,
            reload=not args.no_reload
        )
    finally:
        if pm:
            pm.release_lock()


if __name__ == "__main__":
    run_server()
