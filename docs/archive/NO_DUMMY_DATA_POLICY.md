# NO DUMMY DATA POLICY

## Critical Rule: NEVER USE DUMMY/FAKE DATA

This project MUST NEVER contain dummy, fake, sample, or test data. All data must be either:
1. **Real data from Robinhood API** - live account data, positions, balances
2. **Empty/blank** - show empty states when data is not available
3. **User-entered data** - data manually logged by the user

## What is Prohibited

### ❌ NEVER DO THIS:
- Hardcoded fake balances (e.g., `current_balance = 100000`, `buying_power = 50000`)
- Test trades in the database (e.g., fake NVDA trades for testing)
- Sample positions or placeholder data
- Default values that look like real data
- Mock/dummy returns from API calls

### ✅ ALWAYS DO THIS:
- Check if data exists before displaying
- Show empty states with helpful messages: "Connect to Robinhood to see data"
- Use `0` or `None` as defaults (not fake numbers that look real)
- Pull real data from Robinhood API whenever possible
- Let users manually enter their own real data

## Code Examples

### Bad (NEVER DO THIS):
```python
# ❌ Fake default values
current_balance = account_data.get('portfolio_value', 100000)
buying_power = account_data.get('buying_power', 50000)

# ❌ Test data in database
test_trade = {
    'symbol': 'NVDA',
    'premium': 610.00,
    'strike': 180.00
}
```

### Good (ALWAYS DO THIS):
```python
# ✅ Check for real data first
if account_data:
    current_balance = float(account_data.get('portfolio_value', 0))
    buying_power = float(account_data.get('buying_power', 0))
    # Display metrics...
else:
    st.info("💡 Connect to Robinhood to see your portfolio data")

# ✅ Empty when no data
positions = []  # No dummy data - show empty if not connected
```

## Where This Applies

### Dashboard (`dashboard.py`)
- **Account balances**: Must come from Robinhood or be empty
- **Positions**: Must come from Robinhood or be empty
- **Premium collected**: Calculated from real positions only
- **Trade history**: Only user-logged trades, NO test data

### TradingView Watchlists
- **Current positions**: Real Robinhood positions only
- **Options data**: From database (synced from real APIs)
- **Trade history**: Only real closed trades

### Trade History Database
- **NO test trades**: Delete any NVDA, AAPL, or other test/sample trades
- Only log actual trades the user has made
- Provide manual entry forms for users to log their own trades

### Settings/Configuration
- **NO default example values** that look like real credentials
- Show placeholders in text inputs (these are OK - they're just hints)

## Implementation Checklist

Before committing code, verify:
- [ ] No hardcoded dollar amounts (except 0)
- [ ] No test trades in database
- [ ] All metrics pull from real APIs or show empty
- [ ] Empty states have helpful messages
- [ ] Default values are `0`, `None`, `[]`, or `{}` - not fake data

## Why This Matters

1. **User Trust**: Users must trust the numbers they see
2. **Accuracy**: Trading decisions based on fake data are dangerous
3. **Debugging**: Fake data makes it harder to debug real issues
4. **Professionalism**: Production software should never show fake data

## Recent Violations Fixed

1. ✅ Removed hardcoded `100000` and `50000` fake balances
2. ✅ Deleted fake NVDA test trades from database
3. ✅ Changed defaults from fake numbers to 0 or empty
4. ✅ Added empty state messages when not connected

## Enforcement

**ALL future development must follow this policy.**

If you find dummy data anywhere:
1. Delete it immediately
2. Replace with empty states or Robinhood API calls
3. Document the fix in git commit message
4. Update this policy if needed

## For AI Assistants / Future Developers

When generating code for this project:
- NEVER create test trades or sample data
- NEVER use fake dollar amounts as defaults
- ALWAYS use empty states when data is unavailable
- ALWAYS connect to real Robinhood API when possible
- ALWAYS ask user to manually enter their real data

**This is non-negotiable. Data integrity is critical.**
