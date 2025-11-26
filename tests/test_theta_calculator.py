"""
Test script for theta calculator
"""

from src.theta_calculator import ThetaCalculator
from datetime import datetime, timedelta
import pandas as pd

def test_theta_calculation():
    """Test the theta calculator with sample data"""

    calculator = ThetaCalculator()

    # Test parameters
    current_price = 100.0  # Stock at $100
    strike_price = 95.0    # $95 strike (5% OTM)
    expiration_date = datetime.now() + timedelta(days=30)  # 30 days to expiration
    current_premium = 2.50  # Current option premium
    entry_premium = 3.00    # Entry premium
    implied_volatility = 0.30  # 30% IV
    quantity = 1  # 1 contract

    print("Test Parameters:")
    print(f"Stock Price: ${current_price}")
    print(f"Strike Price: ${strike_price}")
    print(f"Days to Expiration: 30")
    print(f"Entry Premium: ${entry_premium}")
    print(f"Current Premium: ${current_premium}")
    print(f"Implied Volatility: {implied_volatility*100}%")
    print(f"Contracts: {quantity}")
    print()

    # Calculate forecast
    forecast = calculator.calculate_forecast(
        current_price=current_price,
        strike_price=strike_price,
        expiration_date=expiration_date,
        current_premium=current_premium,
        entry_premium=entry_premium,
        implied_volatility=implied_volatility,
        quantity=quantity
    )

    print("Forecast Results:")
    print(f"Max Profit: ${forecast.max_profit:.2f}")
    print(f"Total Expected Decay: ${forecast.total_decay:.2f}")
    print(f"Current P/L: ${forecast.cumulative_pnl[0]:.2f}")
    print(f"Projected P/L at Expiration: ${forecast.cumulative_pnl[-1]:.2f}")
    print()

    # Show first 5 days of forecast
    df = calculator.create_forecast_dataframe(forecast)
    print("First 5 Days of Forecast:")
    print(df.head())
    print()

    # Test Black-Scholes theta directly
    time_to_exp = 30 / 365  # 30 days in years
    theta = calculator.black_scholes_theta(
        S=current_price,
        K=strike_price,
        T=time_to_exp,
        r=calculator.risk_free_rate,
        sigma=implied_volatility,
        option_type='put'
    )

    print(f"Black-Scholes Daily Theta: ${theta:.4f}")
    print(f"Position Theta (100 shares): ${theta * 100:.2f}")

    # Test edge cases
    print("\n--- Edge Case Tests ---")

    # Test expired option
    print("\n1. Expired Option:")
    expired_forecast = calculator.calculate_forecast(
        current_price=current_price,
        strike_price=strike_price,
        expiration_date=datetime.now() - timedelta(days=1),  # Expired yesterday
        current_premium=0,
        entry_premium=entry_premium,
        implied_volatility=implied_volatility,
        quantity=quantity
    )
    print(f"Expired option P/L: ${expired_forecast.cumulative_pnl[0]:.2f}")

    # Test 1 day to expiration
    print("\n2. One Day to Expiration:")
    one_day_forecast = calculator.calculate_forecast(
        current_price=current_price,
        strike_price=strike_price,
        expiration_date=datetime.now() + timedelta(days=1),
        current_premium=0.10,
        entry_premium=entry_premium,
        implied_volatility=implied_volatility,
        quantity=quantity
    )
    print(f"Days remaining: {one_day_forecast.days_remaining[0]}")
    print(f"Theta value: ${one_day_forecast.theta_values[0]:.4f}")

    # Test ITM option
    print("\n3. In-The-Money Option:")
    itm_forecast = calculator.calculate_forecast(
        current_price=90.0,  # Stock below strike
        strike_price=strike_price,
        expiration_date=datetime.now() + timedelta(days=30),
        current_premium=6.00,
        entry_premium=3.00,
        implied_volatility=implied_volatility,
        quantity=quantity
    )
    print(f"ITM Current P/L: ${itm_forecast.cumulative_pnl[0]:.2f}")
    print(f"ITM Projected P/L at Exp: ${itm_forecast.cumulative_pnl[-1]:.2f}")

    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_theta_calculation()