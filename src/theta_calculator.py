import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from typing import List, Tuple
from dataclasses import dataclass
import pandas as pd

@dataclass
class ThetaForecast:
    dates: List[datetime]
    days_remaining: List[int]
    theta_values: List[float]
    option_values: List[float]
    cumulative_pnl: List[float]
    total_decay: float
    max_profit: float

class ThetaCalculator:
    def __init__(self) -> None:
        self.risk_free_rate = 0.05  # 5% risk-free rate

    def black_scholes_theta(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'put') -> float:
        """
        Calculate Black-Scholes theta (time decay per day)

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Implied volatility (as decimal, e.g., 0.30 for 30%)
            option_type: 'call' or 'put'

        Returns:
            Theta per day (divide by 365)
        """
        if T <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'put':
            theta = ((-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                    + r * K * np.exp(-r * T) * norm.cdf(-d2))
        else:  # call
            theta = ((-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                    - r * K * np.exp(-r * T) * norm.cdf(d2))

        # Convert annual theta to daily theta
        daily_theta = theta / 365
        return daily_theta

    def calculate_forecast(self,
                          current_price: float,
                          strike_price: float,
                          expiration_date: datetime,
                          current_premium: float,
                          entry_premium: float,
                          implied_volatility: float,
                          quantity: int = 1,
                          option_type: str = 'put',
                          position_type: str = 'short') -> ThetaForecast:
        """
        Calculate day-by-day theta decay forecast

        Args:
            current_price: Current stock price
            strike_price: Option strike price
            expiration_date: Option expiration date
            current_premium: Current option premium (per share)
            entry_premium: Entry premium when position was opened
            implied_volatility: IV as decimal (e.g., 0.30 for 30%)
            quantity: Number of contracts
            option_type: 'put' or 'call'
            position_type: 'short' or 'long'

        Returns:
            ThetaForecast object with daily projections
        """
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        days_to_exp = (expiration_date - today).days

        if days_to_exp <= 0:
            # Expired
            return ThetaForecast(
                dates=[today],
                days_remaining=[0],
                theta_values=[0],
                option_values=[0],
                cumulative_pnl=[entry_premium * quantity * 100],
                total_decay=entry_premium * quantity * 100,
                max_profit=entry_premium * quantity * 100
            )

        # Generate daily forecasts
        dates = []
        days_remaining = []
        theta_values = []
        option_values = []
        cumulative_pnl = []

        for day in range(days_to_exp + 1):
            forecast_date = today + timedelta(days=day)
            days_left = days_to_exp - day
            time_to_exp = days_left / 365.0  # Convert to years

            # Calculate option value using Black-Scholes
            if days_left > 0:
                d1 = (np.log(current_price / strike_price) +
                     (self.risk_free_rate + 0.5 * implied_volatility ** 2) * time_to_exp) / \
                     (implied_volatility * np.sqrt(time_to_exp))
                d2 = d1 - implied_volatility * np.sqrt(time_to_exp)

                # Calculate option value based on type
                if option_type == 'put':
                    option_value = (strike_price * np.exp(-self.risk_free_rate * time_to_exp) * norm.cdf(-d2) -
                                   current_price * norm.cdf(-d1))
                else:  # call
                    option_value = (current_price * norm.cdf(d1) -
                                   strike_price * np.exp(-self.risk_free_rate * time_to_exp) * norm.cdf(d2))

                # Theta calculation
                theta = self.black_scholes_theta(
                    current_price, strike_price, time_to_exp,
                    self.risk_free_rate, implied_volatility, option_type
                )
            else:
                # At expiration
                if option_type == 'put':
                    option_value = max(strike_price - current_price, 0)
                else:  # call
                    option_value = max(current_price - strike_price, 0)
                theta = 0

            # P/L calculation based on position type
            if position_type == 'short':
                # Short position: profit when option value decreases
                pnl = (entry_premium - option_value) * quantity * 100
            else:  # long
                # Long position: profit when option value increases
                pnl = (option_value - entry_premium) * quantity * 100

            dates.append(forecast_date)
            days_remaining.append(days_left)
            theta_values.append(theta)
            option_values.append(option_value)
            cumulative_pnl.append(pnl)

        total_decay = cumulative_pnl[-1] if cumulative_pnl else 0
        max_profit = entry_premium * quantity * 100

        return ThetaForecast(
            dates=dates,
            days_remaining=days_remaining,
            theta_values=theta_values,
            option_values=option_values,
            cumulative_pnl=cumulative_pnl,
            total_decay=total_decay,
            max_profit=max_profit
        )

    def create_forecast_dataframe(self, forecast: ThetaForecast) -> pd.DataFrame:
        """Convert forecast to pandas DataFrame for display"""
        return pd.DataFrame({
            'Date': forecast.dates,
            'Days Left': forecast.days_remaining,
            'Theta/Day': forecast.theta_values,
            'Option Value': forecast.option_values,
            'Cumulative P/L': forecast.cumulative_pnl
        })