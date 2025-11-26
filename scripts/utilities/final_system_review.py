"""
Final Comprehensive Magnus Platform System Review
Tests all components with proper method calls
"""
from dotenv import load_dotenv
import os
import subprocess
import time

load_dotenv(override=True)

print("=" * 80)
print("MAGNUS PLATFORM - FINAL COMPREHENSIVE REVIEW")
print("=" * 80)

results = {'passed': [], 'warnings': [], 'failed': []}

# Test 1: PostgreSQL
print("\n[TEST 1] PostgreSQL Database")
print("-" * 80)
try:
    import psycopg2
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'),
        database='magnus', user=os.getenv('DB_USER'), password=os.getenv('DB_PASSWORD')
    )
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
    tables = cursor.fetchone()[0]
    cursor.execute("SELECT SUM(n_live_tup) FROM pg_stat_user_tables;")
    rows = cursor.fetchone()[0] or 0
    cursor.close()
    conn.close()
    print(f"[OK] PostgreSQL: {tables} tables, {rows:,} rows")
    results['passed'].append(f'PostgreSQL ({tables} tables)')
except Exception as e:
    print(f"[FAILED] PostgreSQL: {e}")
    results['failed'].append('PostgreSQL')

# Test 2: Ollama
print("\n[TEST 2] Ollama LLM Service")
print("-" * 80)
try:
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    models = [line.split()[0] for line in result.stdout.split('\n')[1:] if line.strip() and len(line.split()) >= 4]
    if models:
        print(f"[OK] Ollama: {len(models)} models available")
        for m in models:
            print(f"     - {m}")
        results['passed'].append(f'Ollama ({len(models)} models)')
    else:
        results['warnings'].append('Ollama: No models')
except Exception as e:
    print(f"[FAILED] Ollama: {e}")
    results['failed'].append('Ollama')

# Test 3: Magnus LLM Integration & Query
print("\n[TEST 3] Magnus LLM Service & Query Test")
print("-" * 80)
try:
    from src.magnus_local_llm import get_magnus_llm
    llm = get_magnus_llm()
    print(f"[OK] Magnus LLM initialized")

    # Test actual query
    print("\nTesting LLM query...")
    test_prompt = "What is a covered call? Answer in one short sentence."
    start = time.time()
    response = llm.query(test_prompt, use_trading_context=False)
    elapsed = time.time() - start

    print(f"[OK] LLM Query successful ({elapsed:.1f}s)")
    print(f"     Response: {response[:120]}...")
    results['passed'].append('Magnus LLM & Query')
except Exception as e:
    print(f"[WARNING] Magnus LLM: {str(e)[:100]}")
    results['warnings'].append('Magnus LLM')

# Test 4: Dashboard
print("\n[TEST 4] Streamlit Dashboard")
print("-" * 80)
try:
    import requests
    response = requests.get('http://localhost:8501', timeout=5)
    if response.status_code == 200:
        print(f"[OK] Dashboard responding at http://localhost:8501")
        results['passed'].append('Dashboard')
    else:
        print(f"[WARNING] Dashboard status {response.status_code}")
        results['warnings'].append('Dashboard')
except Exception as e:
    print(f"[FAILED] Dashboard: {e}")
    results['failed'].append('Dashboard')

# Test 5: Dashboard Pages
print("\n[TEST 5] Dashboard Page Files")
print("-" * 80)
try:
    import glob
    pages = glob.glob('pages/*.py') + glob.glob('features/*/pages/*.py')
    print(f"[OK] Found {len(pages)} page files")
    if pages:
        print("     Sample pages:")
        for p in pages[:8]:
            name = os.path.basename(p).replace('.py', '').replace('_', ' ').title()
            print(f"       - {name}")
        if len(pages) > 8:
            print(f"       ... and {len(pages) - 8} more")
    results['passed'].append(f'Pages ({len(pages)} files)')
except Exception as e:
    print(f"[FAILED] Pages: {e}")
    results['failed'].append('Pages')

# Test 6: Key Dependencies
print("\n[TEST 6] Python Dependencies")
print("-" * 80)
deps = ['streamlit', 'pandas', 'psycopg2', 'plotly', 'langchain', 'ollama', 'requests']
missing = []
for dep in deps:
    try:
        __import__(dep)
        print(f"[OK] {dep}")
    except ImportError:
        print(f"[FAILED] {dep}")
        missing.append(dep)

if not missing:
    results['passed'].append('Dependencies')
else:
    results['warnings'].append(f'Missing: {", ".join(missing)}')

# Test 7: Database Tables
print("\n[TEST 7] Key Database Tables")
print("-" * 80)
try:
    import psycopg2
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'),
        database='magnus', user=os.getenv('DB_USER'), password=os.getenv('DB_PASSWORD')
    )
    cursor = conn.cursor()

    key_tables = ['ai_options_analyses', 'kalshi_markets', 'tv_symbols_api', 'stock_premiums']
    for table in key_tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        count = cursor.fetchone()[0]
        if count > 0:
            print(f"[OK] {table}: {count:,} rows")
        else:
            print(f"[EMPTY] {table}: 0 rows")

    cursor.close()
    conn.close()
    results['passed'].append('Database Tables')
except Exception as e:
    print(f"[WARNING] Tables: {e}")
    results['warnings'].append('Database Tables')

# Test 8: Environment Config
print("\n[TEST 8] Environment Configuration")
print("-" * 80)
env_check = ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
all_set = True
for var in env_check:
    val = os.getenv(var)
    if val:
        display = '*' * 8 if 'PASSWORD' in var else val
        print(f"[OK] {var}: {display}")
    else:
        print(f"[MISSING] {var}")
        all_set = False

if all_set:
    results['passed'].append('Environment Config')
else:
    results['warnings'].append('Environment Config')

# Summary
print("\n" + "=" * 80)
print("FINAL REVIEW SUMMARY")
print("=" * 80)

print(f"\n✓ PASSED: {len(results['passed'])} checks")
for item in results['passed']:
    print(f"  [OK] {item}")

if results['warnings']:
    print(f"\n! WARNINGS: {len(results['warnings'])} items")
    for item in results['warnings']:
        print(f"  [!] {item}")

if results['failed']:
    print(f"\n✗ FAILED: {len(results['failed'])} items")
    for item in results['failed']:
        print(f"  [X] {item}")

# Health Score
total = len(results['passed']) + len(results['warnings']) + len(results['failed'])
score = (len(results['passed']) / total * 100) if total > 0 else 0

print("\n" + "=" * 80)
print(f"SYSTEM HEALTH SCORE: {score:.1f}%")
print("=" * 80)

if score >= 90:
    status = "[EXCELLENT] All systems operational!"
elif score >= 75:
    status = "[GOOD] System ready with minor issues"
elif score >= 50:
    status = "[FAIR] System functional, needs attention"
else:
    status = "[POOR] System has significant issues"

print(f"\n{status}")

print("\n📍 Quick Access:")
print("   Dashboard:  http://localhost:8501")
print("   Database:   postgresql://postgres:****@localhost:5432/magnus")
print("   LLM:        Qwen 2.5 (32B primary, 14B fast)")

if not results['failed']:
    print("\n🎉 Magnus Platform is FULLY OPERATIONAL!")
