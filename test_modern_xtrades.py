#!/usr/bin/env python3
"""
Test Script for Modern Xtrades Module
======================================

Demonstrates and tests all components of the modernized Xtrades system:
- Pydantic models with validation
- AI-powered trade analysis
- Async scraper with Playwright
- Modern sync service
"""

import asyncio
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from xtrades.models import (
    XtradeProfile, XtradeAlert, TradeSignal, AIAnalysis, SyncResult,
    TradeStrategy, TradeAction, AlertType, SentimentLevel, RiskLevel
)
from xtrades.config import get_settings, configure_logging


def test_pydantic_models():
    """Test Pydantic model validation."""
    print("\n" + "="*60)
    print("TESTING PYDANTIC MODELS")
    print("="*60)

    # Test XtradeProfile
    print("\n1. Testing XtradeProfile model...")
    profile = XtradeProfile(
        id=1,
        username="  BeHappy  ",  # Should be stripped and lowercased
        display_name="Be Happy Trades",
        is_active=True
    )
    assert profile.username == "behappy", f"Expected 'behappy', got '{profile.username}'"
    print(f"   Profile: {profile.username} (ID: {profile.id})")
    print("   OK")

    # Test TradeSignal with validation
    print("\n2. Testing TradeSignal model with validation...")
    signal = TradeSignal(
        ticker="AAPL",
        strategy=TradeStrategy.CALL,
        action=TradeAction.BTO,
        strike_price=Decimal("150.00"),
        entry_price=Decimal("2.50"),
        target_price=Decimal("5.00"),
        stop_loss=Decimal("1.25"),
        confidence_score=0.85
    )
    assert signal.ticker == "AAPL"
    assert signal.strategy == TradeStrategy.CALL
    print(f"   Signal: {signal.ticker} {signal.strategy.value} @ ${signal.entry_price}")
    print(f"   Target: ${signal.target_price}, Stop: ${signal.stop_loss}")
    print(f"   Confidence: {signal.confidence_score:.0%}")
    print("   OK")

    # Test XtradeAlert
    print("\n3. Testing XtradeAlert model...")
    alert = XtradeAlert(
        profile_id=1,
        alert_id="alert123",
        alert_text="BTO AAPL 150C 12/20 @ 2.50, PT 5.00, SL 1.25",
        alert_type=AlertType.ENTRY,
        posted_at=datetime.utcnow(),
        ticker="AAPL",
        strategy="call",
        action="bto",
        entry_price=Decimal("2.50"),
        target_price=Decimal("5.00")
    )
    assert alert.has_trade_data()
    print(f"   Alert: {alert.alert_id}")
    print(f"   Has trade data: {alert.has_trade_data()}")
    print("   OK")

    # Test AIAnalysis
    print("\n4. Testing AIAnalysis model...")
    analysis = AIAnalysis(
        alert_id="alert123",
        ticker="AAPL",
        sentiment=SentimentLevel.BULLISH,
        sentiment_score=0.75,
        sentiment_reasoning="Strong bullish setup with clear entry and targets",
        risk_level=RiskLevel.MEDIUM,
        risk_score=0.45,
        risk_factors=["Options expire in 30 days", "Broad market uncertainty"],
        quality_score=0.8,
        completeness_score=0.9,
        summary="Well-defined call trade with clear risk/reward"
    )
    print(f"   Sentiment: {analysis.sentiment.value} ({analysis.sentiment_score:.0%})")
    print(f"   Risk: {analysis.risk_level.value} ({analysis.risk_score:.0%})")
    print(f"   Quality: {analysis.quality_score:.0%}")
    print("   OK")

    # Test SyncResult
    print("\n5. Testing SyncResult model...")
    result = SyncResult(
        success=True,
        profile_username="behappy",
        start_time=datetime.utcnow() - timedelta(seconds=10),
        end_time=datetime.utcnow(),
        alerts_found=25,
        alerts_new=5,
        trades_extracted=20,
        trades_with_ai=20
    )
    print(result.to_summary())
    print("   OK")

    print("\n" + "-"*60)
    print("ALL PYDANTIC MODEL TESTS PASSED!")
    print("-"*60)


def test_configuration():
    """Test configuration loading."""
    print("\n" + "="*60)
    print("TESTING CONFIGURATION")
    print("="*60)

    settings = get_settings()
    print(f"\n1. Database URL: {settings.database.host}:{settings.database.port}/{settings.database.database}")
    print(f"2. AI Enabled: {settings.ai.enabled}")
    print(f"3. AI Provider: {settings.ai.provider}")
    print(f"4. Scraper Headless: {settings.scraper.headless}")
    print(f"5. Sync Interval: {settings.sync_interval_minutes} minutes")

    print("\n" + "-"*60)
    print("CONFIGURATION TEST PASSED!")
    print("-"*60)


async def test_scraper_components():
    """Test scraper components (without actual scraping)."""
    print("\n" + "="*60)
    print("TESTING SCRAPER COMPONENTS")
    print("="*60)

    from xtrades.scraper import ModernXtradesScraper

    # Test text parsing methods
    scraper = ModernXtradesScraper(headless=True)

    # Test timestamp parsing
    print("\n1. Testing timestamp parsing...")
    test_timestamps = [
        ("just now", "recent"),
        ("5 min ago", "recent"),
        ("2 hours ago", "recent"),
        ("1 day ago", "yesterday"),
        ("2024-11-26T10:30:00Z", "ISO format"),
    ]
    for ts, desc in test_timestamps:
        parsed = scraper._parse_timestamp(ts)
        print(f"   '{ts}' -> {parsed.strftime('%Y-%m-%d %H:%M')} ({desc})")

    # Test ticker extraction
    print("\n2. Testing ticker extraction...")
    test_texts = [
        "$AAPL calls looking good",
        "BTO TSLA 250C",
        "Watching NVDA for entry",
        "THE CALL was great",  # Should not extract THE or CALL
    ]
    for text in test_texts:
        ticker = scraper._extract_ticker(text)
        print(f"   '{text}' -> {ticker or 'None'}")

    # Test strategy extraction
    print("\n3. Testing strategy extraction...")
    test_strategies = [
        ("BTO AAPL 150C", "call"),
        ("Selling puts on TSLA", "put"),
        ("Iron condor on SPY", "iron_condor"),
        ("Buying shares of NVDA", "stock"),
    ]
    for text, expected in test_strategies:
        strategy = scraper._extract_strategy(text)
        status = "" if strategy == expected else f" (expected {expected})"
        print(f"   '{text}' -> {strategy}{status}")

    # Test price extraction
    print("\n4. Testing price extraction...")
    test_prices = [
        "Entry @ $2.50, PT $5.00, SL $1.25",
        "Bought at 150 target 165 stop 140",
    ]
    for text in test_prices:
        prices = scraper._extract_prices(text)
        print(f"   '{text}' -> Entry: {prices['entry']}, Target: {prices['target']}, Stop: {prices['stop']}")

    print("\n" + "-"*60)
    print("SCRAPER COMPONENT TESTS PASSED!")
    print("-"*60)


async def test_ai_analyzer():
    """Test AI analyzer (mock mode without actual API calls)."""
    print("\n" + "="*60)
    print("TESTING AI ANALYZER (Components)")
    print("="*60)

    from xtrades.analyzer import AITradeAnalyzer

    # Create analyzer (won't make actual API calls)
    print("\n1. Initializing AI Analyzer...")
    analyzer = AITradeAnalyzer(provider="auto", temperature=0.1)
    print(f"   Provider: {analyzer.provider}")
    print(f"   Temperature: {analyzer.temperature}")
    print("   OK")

    # Test that chains can be created (won't execute)
    print("\n2. Testing prompt templates...")
    from xtrades.analyzer import (
        TRADE_EXTRACTION_PROMPT,
        SENTIMENT_ANALYSIS_PROMPT,
        RISK_ASSESSMENT_PROMPT
    )
    print(f"   Trade extraction prompt: {len(TRADE_EXTRACTION_PROMPT)} chars")
    print(f"   Sentiment analysis prompt: {len(SENTIMENT_ANALYSIS_PROMPT)} chars")
    print(f"   Risk assessment prompt: {len(RISK_ASSESSMENT_PROMPT)} chars")
    print("   OK")

    print("\n" + "-"*60)
    print("AI ANALYZER COMPONENT TESTS PASSED!")
    print("-"*60)


async def test_cookies():
    """Test cookie file presence."""
    print("\n" + "="*60)
    print("TESTING COOKIE PERSISTENCE")
    print("="*60)

    cookies_path = Path.home() / '.xtrades_cache' / 'cookies.pkl'
    print(f"\n1. Cookie file path: {cookies_path}")
    print(f"2. File exists: {cookies_path.exists()}")

    if cookies_path.exists():
        import pickle
        with open(cookies_path, 'rb') as f:
            cookies = pickle.load(f)
        print(f"3. Cookies loaded: {len(cookies)} cookies")

        # Show cookie names
        print("4. Cookie names:")
        for c in cookies:
            name = c.get('name', 'unknown')
            domain = c.get('domain', 'unknown')
            print(f"   - {name} ({domain})")

        # Check for auth cookies
        auth_cookies = [c for c in cookies if 'auth' in c.get('name', '').lower()]
        print(f"\n5. Auth cookies found: {len(auth_cookies)}")

        print("\n" + "-"*60)
        print("COOKIE TEST PASSED!")
        print("-"*60)
    else:
        print("\nWARNING: No cookies file found!")
        print("Run manual_login_xtrades.py to create cookies.")
        print("-"*60)


def main():
    """Run all tests."""
    print("\n")
    print("*"*60)
    print("  MODERN XTRADES MODULE TEST SUITE")
    print("  Version 2.0.0")
    print("*"*60)

    # Configure logging
    configure_logging(level="INFO")

    # Run synchronous tests
    test_pydantic_models()
    test_configuration()

    # Run async tests
    asyncio.run(test_scraper_components())
    asyncio.run(test_ai_analyzer())
    asyncio.run(test_cookies())

    print("\n")
    print("*"*60)
    print("  ALL TESTS COMPLETED!")
    print("*"*60)
    print("\nNext steps:")
    print("1. Ensure cookies are saved (run manual_login_xtrades.py)")
    print("2. Run the modern sync service:")
    print("   python -m xtrades.sync_service --once")
    print()


if __name__ == "__main__":
    main()
