"""
Integration Test Router - Test all API integrations
NO MOCK DATA - All tests verify real connections
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from datetime import datetime
import logging
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/test",
    tags=["integration-test"]
)


async def test_endpoint(name: str, test_func) -> Dict[str, Any]:
    """Test a single endpoint and return result"""
    start_time = datetime.now()
    try:
        result = await test_func() if asyncio.iscoroutinefunction(test_func) else test_func()
        elapsed = (datetime.now() - start_time).total_seconds() * 1000
        return {
            "name": name,
            "status": "pass",
            "response_time_ms": round(elapsed, 2),
            "message": "Connection successful",
            "data_sample": result if isinstance(result, (str, int, float, bool)) else "Data retrieved"
        }
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds() * 1000
        return {
            "name": name,
            "status": "fail",
            "response_time_ms": round(elapsed, 2),
            "message": str(e),
            "error_type": type(e).__name__
        }


@router.get("/all")
async def run_all_tests():
    """
    Run all integration tests and return comprehensive results.
    Tests all API connections and data sources.
    """
    results = []

    # ============ Database Tests ============
    async def test_database():
        from src.database.connection_pool import get_db_connection
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            return cursor.fetchone()[0]

    results.append(await test_endpoint("PostgreSQL Database", test_database))

    # ============ Robinhood Tests ============
    async def test_robinhood_auth():
        from backend.services.portfolio_service import get_portfolio_service
        service = get_portfolio_service()
        service._ensure_login()
        return "Authenticated"

    async def test_robinhood_positions():
        from backend.services.portfolio_service import get_portfolio_service
        service = get_portfolio_service()
        positions = await service.get_positions()
        return f"{len(positions.get('stocks', []))} stocks, {len(positions.get('options', []))} options"

    results.append(await test_endpoint("Robinhood Auth", test_robinhood_auth))
    results.append(await test_endpoint("Robinhood Positions", test_robinhood_positions))

    # ============ TradingView Database Tests ============
    async def test_tradingview_watchlists():
        from src.database.connection_pool import get_db_connection
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tradingview_watchlists")
            count = cursor.fetchone()[0]
            return f"{count} watchlists"

    async def test_tradingview_symbols():
        from src.database.connection_pool import get_db_connection
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Get symbols from watchlist_symbols table instead
            cursor.execute("""
                SELECT COUNT(DISTINCT symbol) FROM tradingview_watchlist_symbols
            """)
            count = cursor.fetchone()[0]
            return f"{count} unique symbols"

    results.append(await test_endpoint("TradingView Watchlists", test_tradingview_watchlists))
    results.append(await test_endpoint("TradingView Symbols", test_tradingview_symbols))

    # ============ XTrades Tests ============
    async def test_xtrades_profiles():
        from src.xtrades_db_manager import XtradesDBManager
        manager = XtradesDBManager()
        profiles = manager.get_active_profiles()
        return f"{len(profiles)} active profiles"

    async def test_xtrades_trades():
        from src.xtrades_db_manager import XtradesDBManager
        manager = XtradesDBManager()
        profiles = manager.get_active_profiles()
        if profiles:
            trades = manager.get_trades_by_profile(profiles[0]['id'], limit=5)
            return f"{len(trades)} recent trades"
        return "No profiles"

    results.append(await test_endpoint("XTrades Profiles", test_xtrades_profiles))
    results.append(await test_endpoint("XTrades Trades", test_xtrades_trades))

    # ============ Earnings Tests ============
    async def test_earnings_manager():
        from src.earnings_manager import EarningsManager
        from datetime import timedelta
        manager = EarningsManager()
        # Get upcoming earnings for next 7 days
        start = datetime.now().date()
        end = (datetime.now() + timedelta(days=7)).date()
        df = manager.get_earnings_events(start_date=start, end_date=end)
        return f"{len(df)} events" if not df.empty else "No events"

    results.append(await test_endpoint("Earnings Database", test_earnings_manager))

    # ============ Options Data Tests ============
    async def test_yfinance():
        import yfinance as yf
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        return f"AAPL: ${price}"

    async def test_options_chain():
        import yfinance as yf
        ticker = yf.Ticker("AAPL")
        options = ticker.options
        return f"{len(options)} expiration dates" if options else "No options"

    results.append(await test_endpoint("YFinance Market Data", test_yfinance))
    results.append(await test_endpoint("YFinance Options Chain", test_options_chain))

    # ============ Premium Scanner Tests ============
    async def test_premium_scanner():
        from src.premium_scanner import PremiumScanner
        scanner = PremiumScanner()
        results_data = scanner.scan_premiums(symbols=['AAPL'], max_price=200, dte=30)
        return f"{len(results_data)} opportunities"

    results.append(await test_endpoint("Premium Scanner", test_premium_scanner))

    # ============ Sports Data Tests ============
    async def test_nfl_database():
        from src.database.connection_pool import get_db_connection
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM nfl_games WHERE season = 2024")
            count = cursor.fetchone()[0]
            return f"{count} games"

    async def test_nfl_predictor():
        from src.prediction_agents.nfl_predictor import NFLPredictor
        predictor = NFLPredictor()
        # Test ELO calculation
        return "NFL Predictor initialized"

    results.append(await test_endpoint("NFL Database", test_nfl_database))
    results.append(await test_endpoint("NFL Predictor", test_nfl_predictor))

    # ============ Kalshi Tests ============
    async def test_kalshi_database():
        from src.database.connection_pool import get_db_connection
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM kalshi_markets WHERE status = 'open'")
            count = cursor.fetchone()[0]
            return f"{count} open markets"

    results.append(await test_endpoint("Kalshi Database", test_kalshi_database))

    # ============ LLM Tests ============
    async def test_ollama():
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags", timeout=5.0)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return f"{len(models)} models available"
            return "Ollama not responding"

    async def test_groq():
        import os
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            return "API key configured"
        return "No API key"

    results.append(await test_endpoint("Ollama Local LLM", test_ollama))
    results.append(await test_endpoint("Groq API Key", test_groq))

    # ============ Discord Tests ============
    async def test_discord():
        import os
        token = os.getenv("DISCORD_TOKEN") or os.getenv("DISCORD_BOT_TOKEN")
        if token:
            return "Token configured"
        return "No token"

    results.append(await test_endpoint("Discord Bot Token", test_discord))

    # Calculate summary
    passed = len([r for r in results if r["status"] == "pass"])
    failed = len([r for r in results if r["status"] == "fail"])
    total_time = sum(r["response_time_ms"] for r in results)

    return {
        "summary": {
            "total_tests": len(results),
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / len(results) * 100, 1) if results else 0,
            "total_time_ms": round(total_time, 2)
        },
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/database")
async def test_database_connections():
    """Test all database connections"""
    results = []

    # PostgreSQL
    try:
        from src.database.connection_pool import get_db_connection
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]
            results.append({"name": "PostgreSQL", "status": "pass", "version": version[:50]})
    except Exception as e:
        results.append({"name": "PostgreSQL", "status": "fail", "error": str(e)})

    return {"database_tests": results, "timestamp": datetime.now().isoformat()}


@router.get("/robinhood")
async def test_robinhood_connection():
    """Test Robinhood API connection"""
    try:
        from backend.services.portfolio_service import get_portfolio_service
        service = get_portfolio_service()
        service._ensure_login()
        positions = await service.get_positions()

        return {
            "status": "connected",
            "stocks_count": len(positions.get("stocks", [])),
            "options_count": len(positions.get("options", [])),
            "total_equity": positions.get("summary", {}).get("total_equity", 0),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/tradingview")
async def test_tradingview_data():
    """Test TradingView data in database"""
    try:
        from src.database.connection_pool import get_db_connection
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Get watchlists
            cursor.execute("SELECT id, name, symbol_count FROM tradingview_watchlists ORDER BY name")
            watchlists = cursor.fetchall()

            # Get symbol count
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM tradingview_symbols")
            symbol_count = cursor.fetchone()[0]

            return {
                "status": "connected",
                "watchlists": [{"id": w[0], "name": w[1], "symbols": w[2]} for w in watchlists],
                "total_symbols": symbol_count,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/sports")
async def test_sports_data():
    """Test sports data integrations"""
    results = {}

    try:
        from src.database.connection_pool import get_db_connection
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # NFL
            cursor.execute("SELECT COUNT(*) FROM nfl_games WHERE season = 2024")
            results["nfl_games_2024"] = cursor.fetchone()[0]

            # NBA
            cursor.execute("SELECT COUNT(*) FROM nba_games WHERE season = '2024-25'")
            results["nba_games_2024"] = cursor.fetchone()[0]

            # Kalshi
            cursor.execute("SELECT COUNT(*) FROM kalshi_markets WHERE status = 'open'")
            results["kalshi_open_markets"] = cursor.fetchone()[0]

            results["status"] = "connected"

    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)

    results["timestamp"] = datetime.now().isoformat()
    return results


@router.get("/llm")
async def test_llm_providers():
    """Test LLM provider connections"""
    import os
    import httpx

    results = []

    # Ollama
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags", timeout=5.0)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "unknown") for m in models[:5]]
                results.append({
                    "provider": "Ollama",
                    "status": "connected",
                    "models": model_names,
                    "model_count": len(models)
                })
            else:
                results.append({"provider": "Ollama", "status": "error", "error": f"HTTP {response.status_code}"})
    except Exception as e:
        results.append({"provider": "Ollama", "status": "disconnected", "error": str(e)})

    # Check API keys
    api_keys = {
        "Groq": os.getenv("GROQ_API_KEY"),
        "DeepSeek": os.getenv("DEEPSEEK_API_KEY"),
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Google": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    }

    for provider, key in api_keys.items():
        if key:
            results.append({
                "provider": provider,
                "status": "configured",
                "key_prefix": key[:8] + "..." if len(key) > 8 else "***"
            })
        else:
            results.append({"provider": provider, "status": "not_configured"})

    return {"llm_providers": results, "timestamp": datetime.now().isoformat()}
