"""Test batch analysis query"""
import sys
sys.path.insert(0, r'c:\code\Magnus')

from src.ai_options_agent.ai_options_db_manager import AIOptionsDBManager

print('Testing database query for options opportunities...\n')

db_manager = AIOptionsDBManager()

# Test fetching opportunities with loose filters
opportunities = db_manager.get_opportunities(
    dte_range=(1, 90),
    delta_range=(-0.50, -0.01),
    min_premium=0,
    limit=10
)

print(f'Found {len(opportunities)} opportunities')
print('\nSample opportunities:')
for opp in opportunities[:5]:
    symbol = opp.get('symbol', 'N/A')
    strike = opp.get('strike_price', 0)
    premium = opp.get('premium', 0)
    dte = opp.get('dte', 0)
    annual_ret = opp.get('annual_return', 0)
    print(f"  {symbol}: Strike ${strike:.2f}, Premium ${premium:.2f}, DTE {dte}, Annual Return {annual_ret:.1f}%")
