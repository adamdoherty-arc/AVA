"""
Quick test script to verify FRED API key is working
"""

from src.economic_indicators import EconomicIndicatorsManager
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

print("=" * 60)
print("FRED API Test")
print("=" * 60)

# Check if API key is loaded
fred_key = os.getenv('FRED_API_KEY')
if fred_key:
    print(f"[OK] FRED API Key loaded from .env: {fred_key[:10]}...")
else:
    print("[ERROR] No FRED API key found in .env")
    exit(1)

# Initialize manager
print("\nInitializing Economic Indicators Manager...")
manager = EconomicIndicatorsManager()

# Test fetching economic snapshot
print("\nFetching economic snapshot...")
try:
    snapshot = manager.get_economic_snapshot()

    print("\n" + "=" * 60)
    print(f"Economic Cycle: {snapshot['cycle']}")
    print("=" * 60)

    print("\nKey Indicators:")
    print(f"  - PMI: {snapshot['pmi']['value']} ({snapshot['pmi']['interpretation']}) - {snapshot['pmi']['trend']}")
    print(f"  - GDP Growth: {snapshot['gdp']['value']}% ({snapshot['gdp']['interpretation']})")
    print(f"  - Unemployment: {snapshot['unemployment']['value']}% ({snapshot['unemployment']['interpretation']}) - {snapshot['unemployment']['trend']}")
    print(f"  - Fed Funds Rate: {snapshot['fed_funds']['value']}% ({snapshot['fed_funds']['interpretation']}) - {snapshot['fed_funds']['trend']}")

    # Get sector recommendations
    print("\nSector Recommendations:")
    recommendations = manager.get_sector_recommendations_from_economy(snapshot)

    print(f"\n  Economic Cycle: {recommendations['cycle']}")
    print(f"  PMI: {recommendations['pmi']}")
    print(f"  Fed Funds: {recommendations['fed_funds']}%")

    print("\n  Overweight Sectors:")
    for sector in recommendations['overweight']:
        print(f"     + {sector}")

    print("\n  Underweight Sectors:")
    for sector in recommendations['underweight']:
        print(f"     - {sector}")

    print("\n" + "=" * 60)
    print("[SUCCESS] FRED API Test Successful!")
    print("=" * 60)

except Exception as e:
    print(f"\n[ERROR] Error fetching data: {e}")
    print("\nThis might be using mock data. Check if FRED API is accessible.")
