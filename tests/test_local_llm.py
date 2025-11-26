"""
Test script for Magnus Local LLM Service
Tests all models and provides performance benchmarks
"""

import time
import sys
from src.magnus_local_llm import MagnusLocalLLM, TaskComplexity, get_magnus_llm

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def test_model_availability():
    """Test if models are available"""
    print_header("MODEL AVAILABILITY CHECK")

    llm = get_magnus_llm()

    for complexity in [TaskComplexity.FAST, TaskComplexity.BALANCED, TaskComplexity.COMPLEX]:
        model = llm.get_model_info(complexity)
        print(f"\n{complexity.value.upper():12} | {model.name:20} | {model.vram_gb:4.1f}GB VRAM | {model.tokens_per_second:3d} tok/s")
        print(f"             Use cases: {', '.join(model.use_cases[:3])}")

def test_simple_query():
    """Test simple query"""
    print_header("SIMPLE QUERY TEST")

    llm = get_magnus_llm()

    prompt = "What is a cash-secured put in options trading?"
    print(f"\nPrompt: {prompt}")
    print("\nResponse (FAST model):")
    print("-" * 70)

    start = time.time()
    response = llm.query(
        prompt=prompt,
        complexity=TaskComplexity.FAST,
        use_trading_context=True
    )
    elapsed = time.time() - start

    print(response)
    print("-" * 70)
    print(f"⏱️  Response time: {elapsed:.2f}s")

def test_trading_analysis():
    """Test trading analysis"""
    print_header("TRADING ANALYSIS TEST")

    llm = get_magnus_llm()

    symbol = "NVDA"
    context = {
        "current_price": 875.50,
        "52_week_high": 950.00,
        "52_week_low": 390.00,
        "volume": 45_000_000,
        "market_cap": "2.15T",
        "pe_ratio": 68.5
    }

    print(f"\nAnalyzing {symbol} with BALANCED model...")
    print(f"Context: {context}")
    print("\nResponse:")
    print("-" * 70)

    start = time.time()
    response = llm.analyze_trade(
        symbol=symbol,
        analysis_type="technical",
        context=context
    )
    elapsed = time.time() - start

    print(response)
    print("-" * 70)
    print(f"⏱️  Response time: {elapsed:.2f}s")

def test_performance_metrics():
    """Test and display performance metrics"""
    print_header("PERFORMANCE METRICS")

    llm = get_magnus_llm()
    metrics = llm.get_metrics()

    print(f"\n{'Metric':<25} {'Value':>15}")
    print("-" * 42)
    print(f"{'Total Requests':<25} {metrics['requests']:>15,}")
    print(f"{'Cache Hits':<25} {metrics['cache_hits']:>15,}")
    print(f"{'Cache Hit Rate':<25} {metrics['cache_hit_rate']:>14.1f}%")
    print(f"{'Cache Size':<25} {metrics['cache_size']:>15,}")
    print(f"{'Errors':<25} {metrics['errors']:>15,}")
    print(f"{'Avg Latency (ms)':<25} {metrics['avg_latency_ms']:>14.0f}")
    print(f"{'Total Tokens':<25} {metrics['total_tokens']:>15,}")

def benchmark_all_models():
    """Benchmark all models"""
    print_header("MODEL BENCHMARK")

    llm = get_magnus_llm()
    test_prompt = "Explain the wheel strategy in trading in 2 sentences."

    print(f"\nBenchmark prompt: \"{test_prompt}\"")
    print("\n{'Model':<20} {'Time (s)':>10} {'Tokens/s (est)':>15}")
    print("-" * 47)

    for complexity in [TaskComplexity.FAST, TaskComplexity.BALANCED]:
        model_name = llm.get_model_info(complexity).name

        try:
            start = time.time()
            response = llm.query(
                prompt=test_prompt,
                complexity=complexity,
                use_trading_context=False
            )
            elapsed = time.time() - start

            # Rough estimate: ~100 words = ~130 tokens
            words = len(response.split())
            estimated_tokens = int(words * 1.3)
            tokens_per_sec = estimated_tokens / elapsed if elapsed > 0 else 0

            print(f"{model_name:<20} {elapsed:>10.2f} {tokens_per_sec:>15.1f}")
        except Exception as e:
            print(f"{model_name:<20} {'ERROR':>10} {str(e)[:20]:>15}")

def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("  MAGNUS LOCAL LLM TEST SUITE")
    print("  NVIDIA RTX 4090 Optimization")
    print("=" * 70)

    tests = [
        ("Model Availability", test_model_availability),
        ("Simple Query", test_simple_query),
        ("Trading Analysis", test_trading_analysis),
        ("Performance Metrics", test_performance_metrics),
        ("Model Benchmark", benchmark_all_models),
    ]

    failed = []

    for test_name, test_func in tests:
        try:
            test_func()
            print(f"\n✓ {test_name} passed")
        except Exception as e:
            print(f"\n✗ {test_name} failed: {e}")
            failed.append((test_name, str(e)))

    # Summary
    print_header("TEST SUMMARY")
    print(f"\nTests run: {len(tests)}")
    print(f"Passed: {len(tests) - len(failed)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed tests:")
        for name, error in failed:
            print(f"  - {name}: {error}")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        print("\nMagnus Local LLM is ready for use!")

if __name__ == "__main__":
    main()
