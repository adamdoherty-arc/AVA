from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
import uvicorn
import logging
import traceback
import sys
import argparse

logger = logging.getLogger(__name__)

# Backend port configuration
BACKEND_PORT = 8002

from backend.routers import (
    sports, predictions, dashboard, chat, research, portfolio, strategy,
    scanner, agents, earnings, xtrades, system, technicals, knowledge,
    enhancements, analytics, options, subscriptions, cache, discord, settings,
    integration_test, watchlist, qa_dashboard,
    # Advanced Technical Analysis routers
    smart_money, advanced_technicals, options_indicators
)

app = FastAPI(
    title="AVA Trading Platform API",
    description="Backend API for AVA - AI-Powered Trading Platform",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler to log all unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Log all unhandled exceptions with full traceback"""
    tb = traceback.format_exc()
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}\n{tb}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "path": request.url.path}
    )

# Core routers
app.include_router(sports.router)
app.include_router(predictions.router)
app.include_router(dashboard.router)
app.include_router(chat.router)
app.include_router(research.router)
app.include_router(portfolio.router)
app.include_router(portfolio.positions_router)  # Alias for /api/positions
app.include_router(strategy.router)
app.include_router(strategy.strategies_router)  # /api/strategies/{symbol} endpoint
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

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AVA Trading Platform API"}

def main():
    """Main entry point with singleton enforcement."""
    parser = argparse.ArgumentParser(description='AVA Backend Server')
    parser.add_argument('--force', action='store_true',
                        help='Force start (kill existing instance)')
    parser.add_argument('--status', action='store_true',
                        help='Show status of backend server')
    parser.add_argument('--stop', action='store_true',
                        help='Stop running backend server')
    parser.add_argument('--port', type=int, default=BACKEND_PORT,
                        help=f'Port to run on (default: {BACKEND_PORT})')
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
    main()
