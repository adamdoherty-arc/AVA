from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import uvicorn
from backend.routers import (
    sports, predictions, dashboard, chat, research, portfolio, strategy,
    scanner, agents, earnings, xtrades, system, technicals, knowledge,
    enhancements, analytics, options, subscriptions, cache, discord, settings,
    integration_test, watchlist,
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

# Core routers
app.include_router(sports.router)
app.include_router(predictions.router)
app.include_router(dashboard.router)
app.include_router(chat.router)
app.include_router(research.router)
app.include_router(portfolio.router)
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

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AVA Trading Platform API"}

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8002, reload=True)
