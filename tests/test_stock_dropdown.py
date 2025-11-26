"""Test stock dropdown functionality"""
import sys
sys.path.insert(0, r'c:\code\Magnus')

from src.components.stock_dropdown import StockDropdown
from src.ai_options_agent.ai_options_db_manager import AIOptionsDBManager

print('Testing Stock Dropdown Component...\n')

# Initialize
db_manager = AIOptionsDBManager()
dropdown = StockDropdown(db_manager)

# Get stock list
stocks_df = dropdown._get_stock_list()

print(f'Found {len(stocks_df)} stocks in database')
print('\nSample stocks (first 10):')
for idx, row in stocks_df.head(10).iterrows():
    formatted = dropdown._format_stock_option(row)
    print(f'  {formatted}')

print('\n' + '='*60)
print(f'Dropdown will show {len(stocks_df)} stocks')
print('Format: SYMBOL - Company Name ($Price)')
print('='*60)
