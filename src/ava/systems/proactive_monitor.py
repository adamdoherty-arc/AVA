"""
Proactive Monitoring System
============================

AVA watches your portfolio 24/7 and tells you what you need to know:
- Morning briefings
- Risk alerts
- Earnings calendar
- Expiring options
- Concentration warnings
- Delta exposure alerts

Author: Magnus Trading Platform
Created: 2025-11-21
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ProactiveMonitor:
    """Proactive monitoring - AVA watches for you!"""

    def __init__(self) -> None:
        """Initialize proactive monitor"""
        self.alert_thresholds = {
            'concentration_max': 0.25,  # 25% in one position
            'sector_max': 0.35,  # 35% in one sector
            'delta_high': 300,  # Net delta warning
            'vix_high': 25,  # High VIX
            'days_to_earnings_warning': 7  # Days before earnings to alert
        }

    def get_risk_alerts(self) -> List[str]:
        """Get all current risk alerts"""
        alerts = []

        try:
            # Check portfolio integration
            from src.services.robinhood_client import get_robinhood_client
            client = get_robinhood_client()

            if not client:
                return ["⚠️ Portfolio not connected"]

            # Get data
            account = client.get_account_info()
            positions = client.get_positions()
            portfolio_value = float(account.get('portfolio_value', 0))

            # Check concentration
            concentration_alerts = self._check_concentration(positions, portfolio_value)
            alerts.extend(concentration_alerts)

            # Check Greeks
            greeks_alerts = self._check_greeks(positions)
            alerts.extend(greeks_alerts)

        except Exception as e:
            logger.error(f"Error getting risk alerts: {e}")
            alerts.append(f"⚠️ Error checking risks: {str(e)[:50]}")

        return alerts

    def _check_concentration(self, positions: List[Dict], portfolio_value: float) -> List[str]:
        """Check for concentration risks"""
        alerts = []

        if not positions or portfolio_value == 0:
            return alerts

        # Check individual positions
        for pos in positions:
            value = float(pos.get('market_value', 0))
            symbol = pos.get('symbol', 'UNKNOWN')
            pct = (value / portfolio_value) if portfolio_value > 0 else 0

            if pct > self.alert_thresholds['concentration_max']:
                alerts.append(f"🚨 {symbol} is {pct*100:.1f}% of portfolio (max: {self.alert_thresholds['concentration_max']*100:.0f}%)")

        return alerts

    def _check_greeks(self, positions: List[Dict]) -> List[str]:
        """Check Greeks exposure"""
        alerts = []

        # Calculate net delta (simplified)
        # In production, sum actual deltas from options positions
        net_delta = 0

        for pos in positions:
            if pos.get('type') == 'option':
                # Simplified: assume delta from contract multiplier
                contracts = int(pos.get('quantity', 0))
                # You'd get actual delta here
                delta_estimate = 0.3  # Placeholder
                net_delta += contracts * 100 * delta_estimate

        if abs(net_delta) > self.alert_thresholds['delta_high']:
            direction = "bullish" if net_delta > 0 else "bearish"
            alerts.append(f"📊 High delta exposure: {net_delta:+.0f} ({direction} bias)")

        return alerts

    def check_earnings_today(self) -> List[str]:
        """Check for earnings today"""
        # In production, integrate with earnings calendar
        # For now, placeholder
        return []  # Would return ['AAPL', 'MSFT', etc.]

    def check_expiring_options(self) -> List[Dict]:
        """Check for options expiring soon"""
        try:
            from src.services.robinhood_client import get_robinhood_client
            client = get_robinhood_client()
            positions = client.get_positions()

            expiring = []
            now = datetime.now()

            for pos in positions:
                if pos.get('type') == 'option':
                    exp_date_str = pos.get('expiration_date')
                    if exp_date_str:
                        exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d')
                        days_to_exp = (exp_date - now).days

                        if 0 <= days_to_exp <= 7:  # Expiring within a week
                            expiring.append({
                                'symbol': pos.get('symbol'),
                                'expiration': exp_date_str,
                                'days': days_to_exp
                            })

            return expiring

        except Exception as e:
            logger.error(f"Error checking expiring options: {e}")
            return []

    def check_ex_dividend_dates(self) -> List[Dict]:
        """Check for upcoming ex-dividend dates"""
        # In production, fetch ex-dividend dates from market data
        return []  # Placeholder
