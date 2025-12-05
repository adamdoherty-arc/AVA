import os
"""
Example Usage of Research API

Demonstrates how to use the FastAPI research endpoints
"""

import asyncio
import requests
from datetime import datetime


# Base URL - uses centralized port 8002
BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8002").rstrip("/api")


def example_get_research(symbol: str):
    """Get research report (uses cache if available)"""
    print(f"\n{'='*60}")
    print(f"Getting research for {symbol}")
    print(f"{'='*60}")

    response = requests.get(f"{BASE_URL}/api/research/{symbol}")

    if response.status_code == 200:
        data = response.json()
        cache_status = response.headers.get('X-Cache', 'UNKNOWN')

        print(f"\nCache Status: {cache_status}")
        print(f"Overall Rating: {data['overall_rating']}/5.0 ⭐")
        print(f"Summary: {data['quick_summary']}\n")

        # Recommendation
        rec = data['analysis']['recommendation']
        print(f"Action: {rec['action']}")
        print(f"Confidence: {rec['confidence']:.0%}")
        print(f"Reasoning: {rec['reasoning']}\n")

        # Scores
        print("Detailed Scores:")
        print(f"  Fundamental: {data['analysis']['fundamental']['score']}/100")
        print(f"  Technical: {data['analysis']['technical']['score']}/100")
        print(f"  Sentiment: {data['analysis']['sentiment']['score']}/100")
        print(f"  Options IV Rank: {data['analysis']['options']['iv_rank']}/100\n")

        # Metadata
        meta = data['metadata']
        print(f"Processing Time: {meta['processing_time_ms']}ms")
        print(f"API Calls: {meta['api_calls_used']}")
        print(f"LLM Model: {meta['llm_model']}")
        print(f"Cache Expires: {meta['cache_expires_at']}")

        return data
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
        return None


def example_force_refresh(symbol: str):
    """Force fresh analysis"""
    print(f"\n{'='*60}")
    print(f"Forcing fresh analysis for {symbol}")
    print(f"{'='*60}")

    response = requests.get(f"{BASE_URL}/api/research/{symbol}/refresh")

    if response.status_code == 200:
        data = response.json()
        print(f"✓ Fresh analysis generated")
        print(f"Overall Rating: {data['overall_rating']}/5.0")
        return data
    else:
        print(f"Error: {response.status_code}")
        return None


def example_check_cache_status(symbol: str):
    """Check cache status without generating analysis"""
    print(f"\n{'='*60}")
    print(f"Checking cache status for {symbol}")
    print(f"{'='*60}")

    response = requests.get(f"{BASE_URL}/api/research/{symbol}/status")

    if response.status_code == 200:
        status = response.json()

        if status['cached']:
            print(f"✓ Cached data available")
            print(f"  Timestamp: {status['timestamp']}")
            print(f"  Age: {status['age_seconds']}s")
            print(f"  Expires in: {status['expires_in_seconds']}s")
            print(f"  Rating: {status['overall_rating']}/5.0")
        else:
            print("✗ No cached data")

        return status
    else:
        print(f"Error: {response.status_code}")
        return None


def example_clear_cache(symbol: str):
    """Clear cached research"""
    print(f"\n{'='*60}")
    print(f"Clearing cache for {symbol}")
    print(f"{'='*60}")

    response = requests.delete(f"{BASE_URL}/api/research/{symbol}/cache")

    if response.status_code == 200:
        data = response.json()
        if data['deleted']:
            print(f"✓ Cache cleared for {symbol}")
        else:
            print(f"No cache found for {symbol}")
        return data
    else:
        print(f"Error: {response.status_code}")
        return None


def example_health_check():
    """Check API health"""
    print(f"\n{'='*60}")
    print(f"Health Check")
    print(f"{'='*60}")

    response = requests.get(f"{BASE_URL}/health")

    if response.status_code == 200:
        health = response.json()
        print(f"Status: {health['status'].upper()}")
        print(f"Timestamp: {health['timestamp']}")
        print("\nComponents:")
        for component, status in health['components'].items():
            emoji = "✓" if status == "healthy" else "✗"
            print(f"  {emoji} {component}: {status}")
        return health
    else:
        print(f"Error: {response.status_code}")
        return None


def example_selective_analysis(symbol: str):
    """Request only specific analysis sections"""
    print(f"\n{'='*60}")
    print(f"Selective analysis for {symbol} (Technical + Options only)")
    print(f"{'='*60}")

    response = requests.get(
        f"{BASE_URL}/api/research/{symbol}",
        params={
            'include_fundamental': False,
            'include_technical': True,
            'include_sentiment': False,
            'include_options': True
        }
    )

    if response.status_code == 200:
        data = response.json()
        print(f"✓ Selective analysis complete")
        print(f"Processing Time: {data['metadata']['processing_time_ms']}ms")
        print(f"Agents Executed: {data['metadata']['agents_executed']}")
        return data
    else:
        print(f"Error: {response.status_code}")
        return None


def example_rate_limit_test():
    """Test rate limiting"""
    print(f"\n{'='*60}")
    print(f"Rate Limit Test (sending 12 requests)")
    print(f"{'='*60}")

    for i in range(12):
        response = requests.get(f"{BASE_URL}/api/research/AAPL")

        if response.status_code == 200:
            print(f"Request {i+1}: ✓ Success")
        elif response.status_code == 429:
            error = response.json()
            print(f"Request {i+1}: ✗ Rate limit exceeded")
            print(f"  Retry after: {error['retry_after_seconds']}s")
            break
        else:
            print(f"Request {i+1}: ✗ Error {response.status_code}")


def example_batch_analysis(symbols: list):
    """Analyze multiple symbols"""
    print(f"\n{'='*60}")
    print(f"Batch Analysis: {', '.join(symbols)}")
    print(f"{'='*60}")

    results = {}

    for symbol in symbols:
        print(f"\nAnalyzing {symbol}...")
        response = requests.get(f"{BASE_URL}/api/research/{symbol}")

        if response.status_code == 200:
            data = response.json()
            results[symbol] = {
                'rating': data['overall_rating'],
                'action': data['analysis']['recommendation']['action'],
                'cached': response.headers.get('X-Cache') == 'HIT'
            }
            print(f"  ✓ Rating: {data['overall_rating']}/5.0 | Action: {data['analysis']['recommendation']['action']}")
        else:
            print(f"  ✗ Failed")
            results[symbol] = None

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for symbol, result in results.items():
        if result:
            cache_emoji = "💾" if result['cached'] else "🔄"
            print(f"{symbol}: {result['rating']:.1f}⭐ | {result['action']} | {cache_emoji}")

    return results


if __name__ == "__main__":
    # Check health first
    example_health_check()

    # Get research for AAPL
    example_get_research("AAPL")

    # Check cache status
    example_check_cache_status("AAPL")

    # Get research again (should hit cache)
    example_get_research("AAPL")

    # Force refresh
    # example_force_refresh("AAPL")

    # Selective analysis
    # example_selective_analysis("MSFT")

    # Clear cache
    # example_clear_cache("AAPL")

    # Batch analysis
    # example_batch_analysis(["AAPL", "MSFT", "GOOGL", "AMZN"])

    # Test rate limiting
    # example_rate_limit_test()
