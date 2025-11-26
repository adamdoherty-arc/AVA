"""
Test script for AI Research Widget
Demonstrates modular usage across different pages
"""

import streamlit as st
from src.components.ai_research_widget import display_consolidated_ai_research_section

st.title("AI Research Widget Test")
st.caption("Testing the modular AI Research component")


# Example 1: Simple usage
st.markdown("### Example 1: Simple Symbol List")
test_symbols_1 = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
display_consolidated_ai_research_section(test_symbols_1, key_prefix="test1")


# Example 2: Different page context
st.markdown("### Example 2: Watchlist Context")
watchlist_symbols = ['AMD', 'INTC', 'MU', 'QCOM']
display_consolidated_ai_research_section(watchlist_symbols, key_prefix="watchlist")


# Example 3: Strategy context
st.markdown("### Example 3: Strategy Context")
strategy_symbols = ['SPY', 'QQQ', 'IWM', 'DIA']
display_consolidated_ai_research_section(strategy_symbols, key_prefix="strategy")

st.success("✅ Widget is fully modular and reusable!")
st.info("""
**Usage Instructions:**

1. Import the widget:
   ```python
   from src.components.ai_research_widget import display_consolidated_ai_research_section
   ```

2. Call it with your symbols and a unique key_prefix:
   ```python
   symbols = ['AAPL', 'MSFT', 'GOOGL']
   display_consolidated_ai_research_section(symbols, key_prefix="my_page")
   ```

3. The widget handles:
   - Deduplication
   - Button generation
   - Session state management
   - Expandable research displays
   - Context isolation with key_prefix
""")
