import sys
import os
import asyncio
import traceback

# Add the project root to the python path
sys.path.append(os.getcwd())

from backend.services.portfolio_service import PortfolioService
from backend.services.research_service import ResearchService
from backend.services.strategy_service import StrategyService

async def test_portfolio():
    print("\n--- Testing PortfolioService ---")
    try:
        service = PortfolioService()
        print("Service initialized.")
        # Try to login (this is usually done in init or first call)
        # The service might need explicit login call if not in init
        # Let's check get_positions
        positions = await service.get_positions()
        print(f"Positions retrieved: {len(positions.get('stocks', [])) + len(positions.get('options', []))}")
    except Exception:
        traceback.print_exc()

async def test_research():
    print("\n--- Testing ResearchService ---")
    try:
        service = ResearchService()
        print("Service initialized.")
        # Test with a symbol
        report = await service.analyze_symbol("AAPL")
        print("Research report retrieved.")
    except Exception:
        traceback.print_exc()

async def test_strategy():
    print("\n--- Testing StrategyService ---")
    try:
        service = StrategyService()
        print("Service initialized.")
        # Test analysis
        results = await service.analyze_watchlist("NVDA")
        print(f"Analysis results: {len(results)}")
    except Exception:
        traceback.print_exc()

async def main():
    print("Starting Debug Session...")
    await test_portfolio()
    await test_research()
    # await test_strategy()
    print("\nDebug Session Complete.")

if __name__ == "__main__":
    asyncio.run(main())
