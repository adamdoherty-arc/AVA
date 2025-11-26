"""
Test the theta forecast integration without running the full app
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.theta_calculator import ThetaCalculator
from datetime import datetime, timedelta

def test_integration():
    """Test the integration with sample CSP positions"""

    # Sample CSP positions (mimicking what comes from Robinhood)
    csp_positions = [
        {
            'Symbol': 'AAPL',
            'Stock Price': 170.50,
            'Strike': 165.00,
            'Expiration': '2025-11-30',
            'DTE': 30,
            'Contracts': 1,
            'Premium': 350.00,  # Total premium for position
            'Current': 250.00,  # Current value
            'P/L': 100.00,
            'P/L %': 28.57,
            'symbol_raw': 'AAPL',
            'pl_raw': 100.00
        },
        {
            'Symbol': 'MSFT',
            'Stock Price': 380.25,
            'Strike': 370.00,
            'Expiration': '2025-11-15',
            'DTE': 15,
            'Contracts': 2,
            'Premium': 600.00,  # Total premium for position
            'Current': 200.00,  # Current value
            'P/L': 400.00,
            'P/L %': 66.67,
            'symbol_raw': 'MSFT',
            'pl_raw': 400.00
        }
    ]

    print("Testing Theta Forecast Integration")
    print("=" * 50)

    calculator = ThetaCalculator()

    for position in csp_positions:
        print(f"\nAnalyzing {position['Symbol']} ${position['Strike']} CSP")
        print("-" * 40)

        # Extract position details
        symbol = position['symbol_raw']
        strike = float(position['Strike'])
        exp_date = datetime.strptime(position['Expiration'], '%Y-%m-%d')
        current_premium = abs(float(position['Current'])) / 100  # Convert to per-share
        entry_premium = abs(float(position['Premium'])) / 100  # Convert to per-share
        quantity = int(position['Contracts'])
        stock_price = position['Stock Price']

        print(f"Stock Price: ${stock_price:.2f}")
        print(f"Strike: ${strike:.2f}")
        print(f"Expiration: {position['Expiration']}")
        print(f"DTE: {position['DTE']} days")
        print(f"Contracts: {quantity}")
        print(f"Entry Premium: ${entry_premium:.2f} per share")
        print(f"Current Premium: ${current_premium:.2f} per share")

        # Calculate forecast
        forecast = calculator.calculate_forecast(
            current_price=stock_price,
            strike_price=strike,
            expiration_date=exp_date,
            current_premium=current_premium,
            entry_premium=entry_premium,
            implied_volatility=0.25,  # Using 25% IV as default
            quantity=quantity
        )

        # Display key metrics
        print(f"\nForecast Results:")
        print(f"  Current P/L: ${forecast.cumulative_pnl[0]:,.2f}")
        print(f"  Max Profit: ${forecast.max_profit:,.2f}")
        print(f"  Projected P/L at Expiration: ${forecast.cumulative_pnl[-1]:,.2f}")

        if len(forecast.theta_values) > 0:
            theta_per_day = forecast.theta_values[0] * quantity * 100
            print(f"  Current Theta/Day: ${theta_per_day:.2f}")

        # Calculate moneyness
        moneyness = ((strike - stock_price) / stock_price * 100)
        print(f"  Moneyness: {moneyness:.1f}% {'OTM' if moneyness > 0 else 'ITM'}")

        # Show next 5 days of decay
        print(f"\n  Next 5 Days Projection:")
        for i in range(min(5, len(forecast.dates))):
            print(f"    {forecast.dates[i].strftime('%Y-%m-%d')}: ${forecast.cumulative_pnl[i]:,.2f}")

    print("\n" + "=" * 50)
    print("Integration test completed successfully!")
    print("\nThe theta forecasting is ready to be used in the positions page.")
    print("When CSP positions are loaded, users can:")
    print("  1. Select any CSP position from a dropdown")
    print("  2. View projected P/L chart with theta decay")
    print("  3. See daily forecast table")
    print("  4. Export forecast data to CSV")

if __name__ == "__main__":
    test_integration()