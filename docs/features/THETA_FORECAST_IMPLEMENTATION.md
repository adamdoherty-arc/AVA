# Theta Decay Forecasting Implementation

## Overview
Successfully implemented a comprehensive theta decay forecasting system for Cash-Secured Put (CSP) positions in the Magnus Trading Platform.

## Files Created/Modified

### 1. **src/theta_calculator.py** (Created)
- Core calculation engine using Black-Scholes model
- Calculates daily theta decay values
- Projects P/L from current date to expiration
- Handles edge cases (expired options, ITM/OTM scenarios)

Key Features:
- `ThetaCalculator` class with configurable risk-free rate (default 5%)
- `black_scholes_theta()`: Calculates theoretical theta per day
- `calculate_forecast()`: Generates day-by-day projections
- `create_forecast_dataframe()`: Converts forecast to pandas DataFrame

### 2. **src/theta_forecast_display.py** (Created)
- UI component for displaying theta forecasts
- Interactive Plotly charts with P/L projections
- Data table with export functionality
- Forecast analysis and insights

Key Features:
- Position selector dropdown
- Real-time IV fetching from options chain (via yfinance)
- Interactive P/L projection chart
- CSV export functionality
- Profit probability assessment based on moneyness

### 3. **positions_page_improved.py** (Modified)
- Added import for theta forecast display
- Integrated `display_theta_forecasts()` after position tables
- Passes CSP positions list to forecast module

## Features Implemented

### 1. Day-by-Day Forecasting
- Projects cumulative P/L from today until expiration
- Shows theta decay value for each day
- Tracks option value changes over time

### 2. Interactive Visualization
- **P/L Projection Chart**: Shows expected profit trajectory
- **Current Position Marker**: Highlights current P/L
- **Max Profit Line**: Shows theoretical maximum gain
- **Break-even Line**: Visual reference at $0

### 3. Key Metrics Display
- Current P/L vs Projected P/L at expiration
- Days to expiration (DTE)
- Current stock price and moneyness
- Implied volatility (fetched or defaulted to 30%)
- Daily theta value in dollars

### 4. Data Export
- Full forecast table viewable in expandable section
- CSV export with all daily projections
- Condensed view for positions with >30 DTE

### 5. Forecast Analysis
- Average daily theta calculation
- Expected total decay to expiration
- Profit probability assessment (High/Medium/Low)
- Risk-free rate transparency

## Technical Implementation

### Black-Scholes Formula
```python
# Put option theta calculation
theta = ((-S * norm.pdf(d1) * sigma) / (2 * sqrt(T))
        + r * K * exp(-r * T) * norm.cdf(-d2)) / 365
```

### P/L Calculation for Short Puts
```python
# For each day until expiration:
pnl = (entry_premium - option_value) * quantity * 100
```

## Testing Results

### Test Cases Verified:
1. **Normal CSP Position**: 30 DTE, 5% OTM - ✅ Working
2. **Expired Option**: Correctly returns max profit - ✅ Working
3. **One Day to Expiration**: Minimal theta, high gamma risk - ✅ Working
4. **ITM Option**: Negative P/L projection - ✅ Working
5. **Multiple Contracts**: Scales correctly - ✅ Working

### Sample Output:
```
AAPL $165 CSP (30 DTE)
- Current P/L: $124.55
- Max Profit: $350.00
- Daily Theta: $-6.37
- Projected gain at expiration: $350.00
```

## User Workflow

1. **Navigate to Positions Page**
   - CSP positions displayed in table format
   - Theta Forecasts section appears below

2. **Select Position**
   - Dropdown shows all CSP positions
   - Format: "SYMBOL $STRIKE exp YYYY-MM-DD"

3. **View Forecast**
   - Interactive chart with P/L projection
   - Key metrics displayed above chart
   - Hover for detailed daily values

4. **Analyze Data**
   - Expand table for day-by-day breakdown
   - Export to CSV for further analysis
   - Review forecast analysis insights

## Dependencies
- **scipy**: For Black-Scholes calculations (norm.pdf, norm.cdf)
- **plotly**: For interactive charting
- **yfinance**: For fetching current stock prices and IV
- **pandas**: For data manipulation
- **numpy**: For mathematical operations

## Edge Cases Handled

1. **Expired Options**: Returns entry premium as final P/L
2. **Missing IV Data**: Defaults to 30% implied volatility
3. **Weekend/Holiday Dates**: Continuous day counting
4. **Large DTE Positions**: Table condensed to every 5th day
5. **API Failures**: Falls back to position data for stock price

## Future Enhancements

1. **Greeks Display**: Add delta, gamma, vega alongside theta
2. **Scenario Analysis**: Multiple stock price paths
3. **Historical Accuracy**: Track forecast vs actual performance
4. **Batch Forecasting**: Aggregate view for all positions
5. **Risk Alerts**: Warnings for rapid theta acceleration
6. **Custom IV Input**: Allow manual IV override
7. **Probability Cones**: Show confidence intervals

## Performance Considerations

- Calculations are performed on-demand (not cached)
- Each forecast generates ~30-45 data points (daily)
- Plotly charts are lightweight and responsive
- CSV exports are generated in-memory

## Validation

The implementation has been tested with:
- Real position data structure
- Various strike/stock price combinations
- Different expiration timeframes
- Multiple contract quantities

All calculations align with standard Black-Scholes model expectations and correctly handle the short put position mechanics (credit received upfront, profit from theta decay).

## Usage Notes

- Theta values are negative (time decay)
- P/L is positive for profitable CSP positions
- Forecast assumes stock price remains constant
- IV is assumed constant throughout forecast period
- Risk-free rate set at 5% (adjustable in code)

## Conclusion

The theta decay forecasting feature is fully implemented and integrated into the positions page. Users can now visualize expected profit trajectories for their CSP positions based on theoretical theta decay, helping them make more informed decisions about position management and exit timing.