"""
Setup World-Class RAG System
Initialize all performance tracking, vector search, and quality scoring
"""
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("WORLD-CLASS RAG SYSTEM SETUP")
print("=" * 80)

print("\n[Step 1/5] Creating performance tracking tables...")
from src.signal_performance_tracker import SignalPerformanceTracker

tracker = SignalPerformanceTracker()
tracker.create_performance_tables()
print("[OK] Created tables:")
print("  - signal_outcomes (track trade results)")
print("  - author_performance (author credibility scores)")
print("  - setup_performance (setup type success rates)")
print("  - signal_quality_scores (composite quality scores)")

print("\n[Step 2/5] Initializing author performance...")
tracker.update_author_performance()
print("[OK] Author performance calculated")

print("\n[Step 3/5] Initializing setup performance...")
tracker.update_setup_performance()
print("[OK] Setup performance calculated")

print("\n[Step 4/5] Calculating quality scores for all signals...")
tracker.calculate_quality_scores()
print("[OK] Quality scores calculated with multi-factor analysis:")
print("  - Author credibility (40% weight)")
print("  - Setup success rate (30% weight)")
print("  - Signal completeness (20% weight)")
print("  - Market alignment (10% weight)")

print("\n[Step 5/5] Indexing signals into ChromaDB for vector search...")
from src.signal_vector_search import SignalVectorSearch

vector_search = SignalVectorSearch()
vector_search.index_all_signals()
print("[OK] Vector embeddings created for semantic similarity search")

print("\n" + "=" * 80)
print("SYSTEM VERIFICATION")
print("=" * 80)

import psycopg2
conn = psycopg2.connect(
    host='localhost',
    port='5432',
    database='magnus',
    user='postgres',
    password=os.getenv('DB_PASSWORD')
)
cur = conn.cursor()

# Check signal quality scores
cur.execute("""
    SELECT
        COUNT(*) as total,
        COUNT(*) FILTER (WHERE recommendation = 'strong_buy') as strong_buy,
        COUNT(*) FILTER (WHERE recommendation = 'buy') as buy,
        COUNT(*) FILTER (WHERE recommendation = 'hold') as hold,
        COUNT(*) FILTER (WHERE recommendation = 'pass') as pass,
        ROUND(AVG(composite_score), 1) as avg_score
    FROM signal_quality_scores
""")
stats = cur.fetchone()

print(f"\nSignal Quality Distribution:")
print(f"  Total signals: {stats[0]}")
print(f"  Strong Buy: {stats[1]} (composite score >= 75)")
print(f"  Buy: {stats[2]} (composite score >= 60)")
print(f"  Hold: {stats[3]} (composite score >= 45)")
print(f"  Pass: {stats[4]} (composite score < 45)")
print(f"  Average composite score: {stats[5]}")

# Check author performance
cur.execute("SELECT COUNT(*) FROM author_performance")
author_count = cur.fetchone()[0]
print(f"\nAuthor performance tracked: {author_count} authors")

# Check setup performance
cur.execute("SELECT COUNT(*) FROM setup_performance")
setup_count = cur.fetchone()[0]
print(f"Setup performance tracked: {setup_count} ticker/setup combinations")

cur.close()
conn.close()

print("\n" + "=" * 80)
print("WORLD-CLASS RAG SYSTEM READY")
print("=" * 80)

print("\n[OK] All components initialized:")
print("  [OK] Performance tracking")
print("  [OK] Author credibility scoring")
print("  [OK] Setup success rate analysis")
print("  [OK] Multi-factor quality scoring")
print("  [OK] Vector search (semantic similarity)")
print("  [OK] AVA query interface")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)

print("\n1. Record trade outcomes using the UI:")
print("   - Go to XTrade Messages > Trading Signals (RAG) tab")
print("   - Click on signals and mark outcomes (win/loss)")
print("   - System will learn and improve recommendations")

print("\n2. AVA can now query using:")
print("   from src.ava_signal_advisor import AVASignalAdvisor")
print("   advisor = AVASignalAdvisor()")
print("   top_5 = advisor.get_top_recommendations(limit=5)")

print("\n3. System will automatically:")
print("   - Track author win rates")
print("   - Identify best setup types per ticker")
print("   - Find similar winning trades")
print("   - Rank signals by probability of success")

print("\n" + "=" * 80)
