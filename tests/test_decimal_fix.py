"""Test decimal conversion fix"""
import sys
sys.path.insert(0, r'c:\code\Magnus')

from decimal import Decimal
from src.ai_options_agent.ai_options_db_manager import AIOptionsDBManager

print("Testing decimal to float conversion...")

# Get sample data from database
db_manager = AIOptionsDBManager()
opportunities = db_manager.get_opportunities(
    symbols=['SOFI'],
    dte_range=(1, 90),
    delta_range=(-0.50, -0.01),
    min_premium=0,
    limit=1
)

if opportunities:
    opp = opportunities[0]
    print(f"\nRaw data from database:")
    print(f"  current_price: {opp.get('current_price')} (type: {type(opp.get('current_price'))})")
    print(f"  strike_price: {opp.get('strike_price')} (type: {type(opp.get('strike_price'))})")

    # Test conversion (same as options_analysis_page.py)
    current_price = float(opp.get('current_price', 0) or opp.get('stock_price', 100))
    print(f"\nConverted current_price: {current_price} (type: {type(current_price)})")

    # Test calculation that was failing
    try:
        price_52w_high = current_price * 1.2
        price_52w_low = current_price * 0.8
        print(f"\n[SUCCESS] - Calculations work:")
        print(f"  52w high estimate: ${price_52w_high:.2f}")
        print(f"  52w low estimate: ${price_52w_low:.2f}")
    except Exception as e:
        print(f"\n[FAILED] - Error: {e}")
else:
    print("No opportunities found for SOFI")
