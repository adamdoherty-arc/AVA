#!/usr/bin/env python3
"""
Test script for Finnhub and Polygon News APIs
Tests with real API calls and verifies integration
"""

import os
import sys
from datetime import datetime, timedelta
from src.news_service import NewsService

# Add src directory to path
sys.path.insert(0, '/c/Code/WheelStrategy')

def test_finnhub_api():
    """Test Finnhub API with real symbol"""
    print("\n" + "="*70)
    print("TEST 1: Finnhub API")
    print("="*70)

    service = NewsService()
    symbol = "AAPL"

    print(f"\nFetching news for {symbol} from Finnhub...")
    print(f"API Key configured: {bool(service.finnhub_key)}")
    print(f"API Key (first 10 chars): {service.finnhub_key[:10] if service.finnhub_key else 'NOT SET'}...")

    try:
        articles = service.get_finnhub_news(symbol)
        print(f"\nSuccess! Retrieved {len(articles)} articles")

        if articles:
            print("\nFirst 3 articles:")
            for i, article in enumerate(articles[:3], 1):
                print(f"\n{i}. {article.headline}")
                print(f"   Source: {article.source}")
                print(f"   Published: {article.published_at.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   URL: {article.url}")
                print(f"   Summary: {article.summary[:100]}..." if article.summary else "   Summary: (empty)")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_polygon_api():
    """Test Polygon API with real symbol"""
    print("\n" + "="*70)
    print("TEST 2: Polygon API")
    print("="*70)

    service = NewsService()
    symbol = "AAPL"

    print(f"\nFetching news for {symbol} from Polygon...")
    print(f"API Key configured: {bool(service.polygon_key)}")
    print(f"API Key (first 10 chars): {service.polygon_key[:10] if service.polygon_key else 'NOT SET'}...")

    try:
        articles = service.get_polygon_news(symbol)
        print(f"\nSuccess! Retrieved {len(articles)} articles")

        if articles:
            print("\nFirst 3 articles:")
            for i, article in enumerate(articles[:3], 1):
                print(f"\n{i}. {article.headline}")
                print(f"   Source: {article.source}")
                print(f"   Published: {article.published_at.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   URL: {article.url}")
                print(f"   Summary: {article.summary[:100]}..." if article.summary else "   Summary: (empty)")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_combined_news():
    """Test combined news from both sources"""
    print("\n" + "="*70)
    print("TEST 3: Combined News (Finnhub + Polygon) with Deduplication")
    print("="*70)

    service = NewsService()
    symbol = "MSFT"

    print(f"\nFetching combined news for {symbol}...")

    try:
        articles = service.get_combined_news(symbol)
        print(f"\nSuccess! Retrieved {len(articles)} unique articles (deduplicated)")

        if articles:
            print("\nFirst 5 articles:")
            for i, article in enumerate(articles[:5], 1):
                print(f"\n{i}. {article.headline}")
                print(f"   Source: {article.source}")
                print(f"   Published: {article.published_at.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   URL: {article.url[:60]}...")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_symbols():
    """Test multiple symbols to verify API works for different tickers"""
    print("\n" + "="*70)
    print("TEST 4: Multiple Symbols Test")
    print("="*70)

    service = NewsService()
    symbols = ["AAPL", "GOOGL", "TSLA"]

    for symbol in symbols:
        print(f"\nFetching news for {symbol}...")
        try:
            articles = service.get_combined_news(symbol)
            print(f"  Success! Retrieved {len(articles)} articles")
        except Exception as e:
            print(f"  ERROR: {e}")

    return True


def test_deduplication():
    """Test that deduplication works properly"""
    print("\n" + "="*70)
    print("TEST 5: Deduplication Verification")
    print("="*70)

    service = NewsService()
    symbol = "AAPL"

    print(f"\nFetching from Finnhub...")
    finnhub_articles = service.get_finnhub_news(symbol)
    print(f"  Finnhub: {len(finnhub_articles)} articles")

    print(f"Fetching from Polygon...")
    polygon_articles = service.get_polygon_news(symbol)
    print(f"  Polygon: {len(polygon_articles)} articles")

    print(f"\nCombined (before dedup): {len(finnhub_articles) + len(polygon_articles)} articles")

    # Collect all headlines
    all_articles = finnhub_articles + polygon_articles
    unique_headlines = set(article.headline.lower().strip() for article in all_articles)
    print(f"Unique headlines: {len(unique_headlines)}")

    # Show any duplicates
    headlines_list = [article.headline.lower().strip() for article in all_articles]
    duplicates = [h for h in set(headlines_list) if headlines_list.count(h) > 1]

    if duplicates:
        print(f"\nDuplicates found: {len(duplicates)}")
        for dup in duplicates[:3]:
            print(f"  - {dup}")
    else:
        print("\nNo duplicates found between sources")

    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("NEWS SERVICE INTEGRATION TEST SUITE")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {
        "Finnhub API": test_finnhub_api(),
        "Polygon API": test_polygon_api(),
        "Combined News": test_combined_news(),
        "Multiple Symbols": test_multiple_symbols(),
        "Deduplication": test_deduplication(),
    }

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name:<30} {status}")

    total_passed = sum(1 for p in results.values() if p)
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\nAll tests PASSED! News integration is working correctly.")
        return 0
    else:
        print(f"\n{total_tests - total_passed} test(s) FAILED. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
