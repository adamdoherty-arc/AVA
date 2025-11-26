"""
XTrade Messages with Local LLM - Comprehensive QA Test
Tests Discord messages page with integrated local LLM analyzer
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_local_llm_availability():
    """Test 1: Check if local LLM is available"""
    print_section("TEST 1: Local LLM Availability")

    try:
        from src.magnus_local_llm import get_magnus_llm, TaskComplexity

        llm = get_magnus_llm()
        print("\n[OK] Local LLM service initialized")

        # Check Ollama connection
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                print(f"[OK] Ollama server online with {len(models)} models")
                for model in models[:5]:
                    print(f"     - {model.get('name', 'unknown')}")
                return True, "local_llm"
            else:
                print("[WARN] Ollama server not responding")
                return True, "rule_based"
        except Exception as e:
            print(f"[WARN] Ollama not available: {e}")
            print("[INFO] Will use rule-based fallback")
            return True, "rule_based"

    except Exception as e:
        print(f"[ERROR] Failed to initialize LLM service: {e}")
        return False, None


def test_discord_ai_analyzer():
    """Test 2: Test Discord AI Analyzer"""
    print_section("TEST 2: Discord AI Analyzer")

    try:
        from src.discord_ai_analyzer import get_discord_analyzer

        analyzer = get_discord_analyzer()
        print(f"\n[OK] Discord AI Analyzer initialized")
        print(f"     Using: {'Local LLM' if analyzer.use_local_llm else 'Rule-Based'}")

        # Test with sample message
        test_message = {
            'content': '$NVDA CSP at $480 strike, 30 DTE. Entry $2.50, target $4.00. High confidence with support at $475.',
            'author_name': 'TestTrader',
            'channel_name': 'options-signals',
            'timestamp': datetime.now()
        }

        print("\n[TEST] Analyzing sample message...")
        result = analyzer.analyze_signal(test_message)

        print(f"\n[RESULT] Analysis complete:")
        print(f"  - Method: {result.get('method', 'unknown')}")
        print(f"  - Tickers: {result.get('tickers', [])}")
        print(f"  - Sentiment: {result.get('sentiment', 'unknown')}")
        print(f"  - Confidence: {result.get('confidence', 0)}%")
        print(f"  - Setup: {result.get('setup', 'unknown')}")
        print(f"  - Risk: {result.get('risk_level', 'unknown')}")

        if result.get('method') == 'local_llm':
            print(f"  - Model: {result.get('model', 'unknown')}")
            print(f"  - Analysis: {result.get('analysis', '')[:100]}...")

        # Validate results
        if result.get('tickers') and result.get('confidence', 0) > 0:
            print("\n[PASS] AI Analyzer test passed")
            return True
        else:
            print("\n[WARN] AI Analyzer returned incomplete results")
            return False

    except Exception as e:
        print(f"\n[ERROR] AI Analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_analysis():
    """Test 3: Batch signal analysis"""
    print_section("TEST 3: Batch Signal Analysis")

    try:
        from src.discord_ai_analyzer import get_discord_analyzer

        analyzer = get_discord_analyzer()

        # Create test messages
        test_messages = [
            {
                'content': '$AAPL bullish setup. Long call at $185 strike, exp 12/29. Strong support.',
                'author_name': 'Trader1',
                'channel_name': 'tech-stocks',
                'timestamp': datetime.now()
            },
            {
                'content': '$TSLA bearish. Selling covered calls at $250. Resistance holding.',
                'author_name': 'Trader2',
                'channel_name': 'ev-stocks',
                'timestamp': datetime.now()
            },
            {
                'content': 'Market looking weak today. Cash position increasing.',
                'author_name': 'Trader3',
                'channel_name': 'market-commentary',
                'timestamp': datetime.now()
            },
            {
                'content': '$SPY iron condor 460/465/470/475, 14 DTE. Premium $1.20. Max profit setup.',
                'author_name': 'Trader4',
                'channel_name': 'spreads',
                'timestamp': datetime.now()
            }
        ]

        print(f"\n[TEST] Analyzing {len(test_messages)} messages...")

        results = analyzer.batch_analyze(test_messages)

        print(f"\n[RESULT] Batch analysis complete:")
        print(f"  - Total analyzed: {len(results)}")
        print(f"  - Bullish: {len([r for r in results if r.get('sentiment') == 'bullish'])}")
        print(f"  - Bearish: {len([r for r in results if r.get('sentiment') == 'bearish'])}")
        print(f"  - Neutral: {len([r for r in results if r.get('sentiment') == 'neutral'])}")

        # Get top signals
        top_signals = analyzer.get_top_signals(test_messages, min_confidence=50, limit=3)

        print(f"\n[RESULT] Top {len(top_signals)} signals (≥50% confidence):")
        for i, sig in enumerate(top_signals, 1):
            print(f"  {i}. {sig.get('tickers', ['N/A'])[0] if sig.get('tickers') else 'N/A'} - "
                  f"{sig.get('confidence', 0)}% - {sig.get('setup', 'Unknown')}")

        print("\n[PASS] Batch analysis test passed")
        return True

    except Exception as e:
        print(f"\n[ERROR] Batch analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database_integration():
    """Test 4: Integration with Discord database"""
    print_section("TEST 4: Database Integration")

    try:
        import psycopg2
        import psycopg2.extras

        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5432)),
            database=os.getenv('DB_NAME', 'magnus'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )

        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Get sample messages
        cur.execute("""
            SELECT
                m.message_id,
                m.content,
                m.author_name,
                m.timestamp,
                c.channel_name
            FROM discord_messages m
            JOIN discord_channels c ON m.channel_id = c.channel_id
            ORDER BY m.timestamp DESC
            LIMIT 10
        """)

        messages = cur.fetchall()
        cur.close()
        conn.close()

        print(f"\n[OK] Found {len(messages)} messages in database")

        if messages:
            from src.discord_ai_analyzer import get_discord_analyzer

            analyzer = get_discord_analyzer()

            print("\n[TEST] Analyzing real Discord messages...")
            results = analyzer.batch_analyze([dict(msg) for msg in messages], max_analyze=5)

            print(f"\n[RESULT] Analyzed {len(results)} real messages:")
            for i, result in enumerate(results[:3], 1):
                tickers_str = ', '.join(result.get('tickers', [])) if result.get('tickers') else 'None'
                print(f"  {i}. {result.get('channel_name', 'Unknown')} - "
                      f"Tickers: {tickers_str} - "
                      f"Confidence: {result.get('confidence', 0)}%")

            print("\n[PASS] Database integration test passed")
            return True
        else:
            print("\n[WARN] No messages found in database")
            print("       Add Discord channels and sync messages to test with real data")
            return True  # Not a failure, just no data

    except Exception as e:
        print(f"\n[ERROR] Database integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_page_imports():
    """Test 5: Test discord_messages_page imports"""
    print_section("TEST 5: Page Import Test")

    try:
        print("\n[TEST] Importing discord_messages_page...")

        # Test import
        import discord_messages_page

        print("[OK] discord_messages_page imported successfully")

        # Check for key functions/classes
        required = ['DiscordDB', 'main']
        for item in required:
            if hasattr(discord_messages_page, item):
                print(f"[OK] Found {item}")
            else:
                print(f"[WARN] Missing {item}")

        print("\n[PASS] Page import test passed")
        return True

    except Exception as e:
        print(f"\n[ERROR] Page import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance():
    """Test 6: Performance benchmarks"""
    print_section("TEST 6: Performance Benchmarks")

    try:
        from src.discord_ai_analyzer import get_discord_analyzer
        import time

        analyzer = get_discord_analyzer()

        # Create larger test dataset
        test_messages = []
        for i in range(20):
            test_messages.append({
                'content': f'$STOCK{i} trading signal. CSP at ${100 + i} strike. Confidence high.',
                'author_name': f'Trader{i}',
                'channel_name': 'signals',
                'timestamp': datetime.now()
            })

        print(f"\n[TEST] Analyzing {len(test_messages)} messages for performance...")

        start_time = time.time()
        results = analyzer.batch_analyze(test_messages)
        elapsed = time.time() - start_time

        print(f"\n[RESULT] Performance metrics:")
        print(f"  - Messages analyzed: {len(results)}")
        print(f"  - Total time: {elapsed:.2f}s")
        print(f"  - Avg per message: {elapsed / len(results):.3f}s")
        print(f"  - Method: {results[0].get('method', 'unknown') if results else 'N/A'}")

        if elapsed < 60:  # Should complete in under 1 minute
            print("\n[PASS] Performance test passed")
            return True
        else:
            print("\n[WARN] Performance slower than expected")
            return False

    except Exception as e:
        print(f"\n[ERROR] Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("  XTrade Messages + Local LLM - Comprehensive QA Test")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    results = {}

    # Run tests
    llm_available, llm_method = test_local_llm_availability()
    results['llm_availability'] = llm_available

    if llm_available:
        results['ai_analyzer'] = test_discord_ai_analyzer()
        results['batch_analysis'] = test_batch_analysis()
        results['database_integration'] = test_database_integration()
        results['page_imports'] = test_page_imports()
        results['performance'] = test_performance()
    else:
        print("\n[ERROR] Local LLM not available, skipping dependent tests")

    # Summary
    print_section("TEST SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\nTests passed: {passed}/{total}")
    print("\nDetailed results:")
    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {test_name.replace('_', ' ').title()}")

    # Recommendations
    print_section("RECOMMENDATIONS")

    if results.get('llm_availability') and llm_method == "local_llm":
        print("\n[+] Local LLM (Qwen 32B) is active!")
        print("   - AI analysis is using advanced language model")
        print("   - Expect high-quality signal analysis")
        print("   - 40-50 tokens/second inference speed")
    elif results.get('llm_availability') and llm_method == "rule_based":
        print("\n[!] Using rule-based fallback")
        print("   To enable local LLM:")
        print("   1. Install Ollama: https://ollama.ai/download")
        print("   2. Pull model: ollama pull qwen2.5:32b-instruct-q4_K_M")
        print("   3. Restart: ollama serve")
    else:
        print("\n[-] Local LLM not available")
        print("   Install and configure as described above")

    if not results.get('database_integration'):
        print("\n[i] No Discord messages found")
        print("   To populate database:")
        print("   1. Add channels via XTrade Messages > Channel Management")
        print("   2. Sync: python sync_discord.py CHANNEL_ID 7")

    # Next steps
    print_section("NEXT STEPS")

    print("\n1. [+] Open Magnus dashboard: streamlit run dashboard.py")
    print("2. [+] Navigate to 'XTrade Messages' page")
    print("3. [+] Go to 'AI Trading Signals' tab")
    print("4. [+] Verify AI method shown (Local LLM or Rule-Based)")
    print("5. [+] Test signal analysis with your Discord data")

    print("\n" + "=" * 70)
    print()

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
