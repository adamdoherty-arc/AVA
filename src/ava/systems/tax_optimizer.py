"""
Tax Optimization Agent
======================

Save money on taxes legally:
- Tax-loss harvesting opportunities
- Wash sale warnings
- Asset location optimization
- Long-term vs short-term capital gains
- Tax-efficient trading strategies

Author: Magnus Trading Platform
Created: 2025-11-21
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TaxOptimizer:
    """Intelligent tax optimization"""

    def __init__(self) -> None:
        """Initialize tax optimizer"""
        self.wash_sale_days = 30  # 30 days before and after
        self.long_term_threshold = 365  # Days to qualify for long-term gains

    def find_opportunities(self, positions: List[Dict]) -> List[Dict]:
        """
        Find tax optimization opportunities.

        Args:
            positions: List of portfolio positions

        Returns:
            List of tax-saving opportunities
        """
        opportunities = []

        # Tax-loss harvesting
        tlh_opps = self._find_tax_loss_harvesting(positions)
        opportunities.extend(tlh_opps)

        # Wash sale warnings
        wash_warnings = self._check_wash_sales(positions)
        opportunities.extend(wash_warnings)

        # Long-term vs short-term
        holding_period_opps = self._analyze_holding_periods(positions)
        opportunities.extend(holding_period_opps)

        return opportunities

    def _find_tax_loss_harvesting(self, positions: List[Dict]) -> List[Dict]:
        """Find tax-loss harvesting opportunities"""
        opportunities = []

        for pos in positions:
            # Check if position is at a loss
            entry_price = float(pos.get('average_buy_price', 0))
            current_price = float(pos.get('current_price', 0))

            if current_price < entry_price:
                loss = (entry_price - current_price) * float(pos.get('quantity', 0))
                loss_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

                if loss_pct < -5:  # More than 5% loss
                    opportunities.append({
                        'type': 'tax_loss_harvesting',
                        'symbol': pos.get('symbol'),
                        'loss_amount': loss,
                        'loss_pct': loss_pct,
                        'recommendation': f"Sell {pos.get('symbol')} to realize ${abs(loss):,.2f} loss for tax offset",
                        'savings_estimate': abs(loss) * 0.25,  # Estimate 25% tax rate
                        'priority': 'high' if loss_pct < -10 else 'medium'
                    })

        return opportunities

    def _check_wash_sales(self, positions: List[Dict]) -> List[Dict]:
        """Check for potential wash sale violations"""
        warnings = []

        # In production, track recent sales and warn about repurchases
        # For now, placeholder

        return warnings

    def _analyze_holding_periods(self, positions: List[Dict]) -> List[Dict]:
        """Analyze holding periods for tax efficiency"""
        opportunities = []

        for pos in positions:
            # Check how long held
            purchase_date_str = pos.get('created_at')
            if not purchase_date_str:
                continue

            try:
                purchase_date = datetime.fromisoformat(purchase_date_str.replace('Z', '+00:00'))
                days_held = (datetime.now(purchase_date.tzinfo) - purchase_date).days

                # Close to long-term threshold
                if self.long_term_threshold - 30 <= days_held < self.long_term_threshold:
                    days_remaining = self.long_term_threshold - days_held

                    opportunities.append({
                        'type': 'holding_period',
                        'symbol': pos.get('symbol'),
                        'days_held': days_held,
                        'days_to_long_term': days_remaining,
                        'recommendation': f"Wait {days_remaining} days to sell {pos.get('symbol')} for long-term capital gains rate",
                        'tax_savings_estimate': float(pos.get('unrealized_plnl', 0)) * 0.1,  # 10% difference between short/long term
                        'priority': 'medium'
                    })

            except Exception as e:
                logger.warning(f"Error parsing date for {pos.get('symbol')}: {e}")
                continue

        return opportunities

    def calculate_tax_impact(
        self,
        realized_gains: float,
        realized_losses: float,
        holding_period: str = 'short'  # 'short' or 'long'
    ) -> Dict:
        """
        Calculate estimated tax impact.

        Args:
            realized_gains: Total realized gains
            realized_losses: Total realized losses
            holding_period: 'short' or 'long' term

        Returns:
            Tax impact estimate
        """
        net_gain = realized_gains - realized_losses

        # Tax rates (estimates - actual depends on income bracket)
        if holding_period == 'long':
            tax_rate = 0.15  # 15% long-term capital gains
        else:
            tax_rate = 0.25  # 25% short-term (ordinary income)

        tax_owed = max(0, net_gain * tax_rate)

        return {
            'realized_gains': realized_gains,
            'realized_losses': realized_losses,
            'net_gain': net_gain,
            'tax_rate': tax_rate,
            'estimated_tax': tax_owed,
            'holding_period': holding_period
        }
