"""
Test Discord Signal Extraction on Existing Messages
"""
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("TESTING DISCORD SIGNAL EXTRACTION")
print("=" * 80)

# Import extractor
from src.discord_signal_extractor import DiscordSignalExtractor

print("\n[1/3] Initializing extractor...")
extractor = DiscordSignalExtractor()
print("[OK] Extractor initialized")

print("\n[2/3] Processing all Discord messages...")
print("This will:")
print("  - Read all messages from discord_messages table")
print("  - Extract tickers, prices, option info, sentiment")
print("  - Store structured data in discord_trading_signals table")
print("  - Calculate confidence scores")
print()

try:
    count = extractor.process_all_messages()
    print(f"\n[OK] Successfully extracted {count} trading signals!")
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n[3/3] Verifying extracted signals...")

import psycopg2
import psycopg2.extras

conn = psycopg2.connect(
    host='localhost',
    port='5432',
    database='magnus',
    user='postgres',
    password=os.getenv('DB_PASSWORD')
)
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# Total signals
cur.execute("SELECT COUNT(*) as count FROM discord_trading_signals")
total = cur.fetchone()['count']
print(f"\nTotal signals in database: {total}")

# Signals by confidence
cur.execute("""
    SELECT
        CASE
            WHEN confidence >= 75 THEN '75-100% (High)'
            WHEN confidence >= 60 THEN '60-74% (Medium)'
            WHEN confidence >= 40 THEN '40-59% (Low)'
            ELSE '0-39% (Very Low)'
        END as confidence_range,
        COUNT(*) as count
    FROM discord_trading_signals
    GROUP BY confidence_range
    ORDER BY MIN(confidence) DESC
""")
print("\nSignals by confidence:")
for row in cur.fetchall():
    print(f"  {row['confidence_range']}: {row['count']}")

# Signals by sentiment
cur.execute("""
    SELECT sentiment, COUNT(*) as count
    FROM discord_trading_signals
    GROUP BY sentiment
    ORDER BY count DESC
""")
print("\nSignals by sentiment:")
for row in cur.fetchall():
    print(f"  {row['sentiment'].title()}: {row['count']}")

# Signals by setup type
cur.execute("""
    SELECT setup_type, COUNT(*) as count
    FROM discord_trading_signals
    GROUP BY setup_type
    ORDER BY count DESC
    LIMIT 5
""")
print("\nTop 5 setup types:")
for row in cur.fetchall():
    print(f"  {row['setup_type'].replace('_', ' ').title()}: {row['count']}")

# Top tickers
cur.execute("""
    SELECT primary_ticker, COUNT(*) as count
    FROM discord_trading_signals
    WHERE primary_ticker IS NOT NULL
    GROUP BY primary_ticker
    ORDER BY count DESC
    LIMIT 10
""")
print("\nTop 10 tickers:")
for row in cur.fetchall():
    print(f"  {row['primary_ticker']}: {row['count']} signals")

# Sample high confidence signals
cur.execute("""
    SELECT
        primary_ticker,
        setup_type,
        sentiment,
        entry,
        target,
        confidence,
        LEFT(content, 80) as preview
    FROM discord_trading_signals
    WHERE confidence >= 70
    ORDER BY confidence DESC, timestamp DESC
    LIMIT 5
""")
print("\n" + "=" * 80)
print("SAMPLE HIGH CONFIDENCE SIGNALS (>=70%)")
print("=" * 80)
high_conf = cur.fetchall()
if high_conf:
    for i, signal in enumerate(high_conf, 1):
        print(f"\n[{i}] {signal['primary_ticker']} - {signal['confidence']}% confidence")
        print(f"    Setup: {signal['setup_type'].replace('_', ' ').title()}")
        print(f"    Sentiment: {signal['sentiment'].title()}")
        if signal['entry']:
            print(f"    Entry: ${signal['entry']:.2f}")
        if signal['target']:
            print(f"    Target: ${signal['target']:.2f}")
        print(f"    Preview: {signal['preview']}...")
else:
    print("\nNo high confidence signals found (>=70%)")

cur.close()
conn.close()

print("\n" + "=" * 80)
print("EXTRACTION COMPLETE")
print("=" * 80)
print("\n[OK] All signals are now stored in discord_trading_signals table")
print("[OK] Ready for AVA to query via RAG system")
print("[OK] View in Streamlit: XTrade Messages -> Trading Signals (RAG) tab")
print("\n" + "=" * 80)
