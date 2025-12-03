"""
CSP Recovery Analyzer - Intelligent recommendations for recovering losing CSP positions

This module analyzes losing Cash Secured Put positions and recommends optimal
recovery strategies including:
- Lower strike CSPs with better premiums
- Optimal entry points based on support levels
- Risk-adjusted scoring and ranking
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


class CSPRecoveryAnalyzer:
    """Analyzes losing CSP positions and finds recovery opportunities"""

    def __init__(self) -> None:
        self.risk_free_rate = 0.045  # Current risk-free rate
        self.trading_days = 252

    def analyze_losing_positions(self, positions: List[Dict]) -> List[Dict]:
        """
        Analyze current losing CSP positions

        Args:
            positions: List of current option positions (already filtered for losing positions)

        Returns:
            List of analyzed losing positions with metrics
        """
        losing_positions = []

        for position in positions:
            # Positions passed in are already identified as losing based on P/L
            # We just need to check if they are puts and short
            if position.get('option_type', '').lower() == 'put' and position.get('position_type') == 'short':
                # Get current stock price
                current_price = self._get_current_price(position['symbol'])
                strike_price = float(position.get('strike_price', 0))

                # Position is already identified as losing - calculate loss details
                # Use the actual loss from position data if available
                if position.get('current_loss') is not None:
                    loss_amount = position.get('current_loss')
                    loss_percentage = position.get('loss_percentage', 0)
                else:
                    # Calculate based on stock price vs strike
                    if current_price < strike_price:
                        loss_amount = (strike_price - current_price) * 100 * abs(position.get('quantity', 1))
                        loss_percentage = ((strike_price - current_price) / strike_price) * 100
                    else:
                        # Stock above strike but position losing due to premium increase
                        loss_amount = position.get('current_loss', 0)
                        loss_percentage = position.get('loss_percentage', 0)

                position_analysis = {
                    'symbol': position['symbol'],
                    'current_strike': strike_price,
                    'current_price': current_price,
                    'expiration': position.get('expiration_date'),
                    'premium_collected': float(position.get('average_price', 0)) * 100 if position.get('average_price') else position.get('premium_collected', 0),
                    'current_loss': loss_amount,
                    'loss_percentage': loss_percentage,
                    'days_to_expiry': self._calculate_days_to_expiry(position.get('expiration_date')),
                    'quantity': abs(position.get('quantity', 1))
                }

                # Add technical levels
                support_levels = self._calculate_support_levels(position['symbol'])
                position_analysis['support_levels'] = support_levels

                # Add IV rank
                position_analysis['iv_rank'] = self._calculate_iv_rank(position['symbol'])

                losing_positions.append(position_analysis)

        return losing_positions

    def find_recovery_opportunities(self, position: Dict, num_strikes: int = 5) -> List[Dict]:
        """
        Find optimal recovery CSP opportunities for a losing position

        Args:
            position: Current losing position details
            num_strikes: Number of strike recommendations to generate

        Returns:
            List of recovery opportunities with scoring
        """
        symbol = position['symbol']
        current_price = position['current_price']
        current_strike = position['current_strike']

        opportunities = []

        # Get option chain
        try:
            ticker = yf.Ticker(symbol)
            exp_dates = ticker.options[:3]  # Next 3 expiration dates

            for exp_date in exp_dates:
                option_chain = ticker.option_chain(exp_date)
                puts = option_chain.puts

                # Filter for strikes below current price
                recovery_strikes = puts[puts['strike'] < current_price * 0.95]

                if recovery_strikes.empty:
                    continue

                # Sort by premium yield
                recovery_strikes['yield'] = (recovery_strikes['bid'] / recovery_strikes['strike']) * 100
                recovery_strikes = recovery_strikes.nlargest(num_strikes, 'yield')

                for _, opt in recovery_strikes.iterrows():
                    opportunity = self._evaluate_recovery_option(
                        symbol=symbol,
                        current_price=current_price,
                        strike=opt['strike'],
                        premium=opt['bid'],
                        expiration=exp_date,
                        current_position=position
                    )
                    opportunities.append(opportunity)

        except Exception as e:
            logger.error(f"Error getting option chain for {symbol}: {e}")

        # Rank opportunities by AI score
        opportunities = self.rank_by_score(opportunities)

        return opportunities[:num_strikes]

    def calculate_recovery_metrics(self, current_pos: Dict, new_strike: float,
                                  new_premium: float, expiration: str) -> Dict:
        """
        Calculate detailed recovery metrics for a potential new position

        Args:
            current_pos: Current losing position
            new_strike: Proposed new strike price
            new_premium: Premium for new strike
            expiration: Expiration date

        Returns:
            Dictionary of recovery metrics
        """
        current_price = current_pos['current_price']
        days_to_expiry = self._calculate_days_to_expiry(expiration)

        # Calculate probability of profit (stock staying above strike)
        volatility = self._calculate_volatility(current_pos['symbol'])
        prob_profit = self._calculate_probability_otm(
            current_price, new_strike, volatility, days_to_expiry
        )

        # Calculate annualized yield
        premium_yield = (new_premium / new_strike) * 100
        annualized_yield = (premium_yield * 365 / days_to_expiry) if days_to_expiry > 0 else 0

        # Calculate breakeven
        breakeven_price = new_strike - new_premium

        # Calculate max profit and loss
        max_profit = new_premium * 100
        max_loss = (new_strike - new_premium) * 100

        # Calculate recovery potential
        recovery_amount = new_premium * 100
        recovery_percentage = (recovery_amount / abs(current_pos['current_loss'])) * 100 if current_pos['current_loss'] != 0 else 0

        return {
            'strike': new_strike,
            'premium': new_premium,
            'yield_percent': premium_yield,
            'annualized_yield': annualized_yield,
            'probability_profit': prob_profit,
            'breakeven': breakeven_price,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'days_to_expiry': days_to_expiry,
            'expiration': expiration,
            'recovery_amount': recovery_amount,
            'recovery_percentage': recovery_percentage,
            'margin_required': new_strike * 100  # Cash secured
        }

    def rank_by_score(self, opportunities: List[Dict]) -> List[Dict]:
        """
        Rank recovery opportunities by composite AI score

        Args:
            opportunities: List of recovery opportunities

        Returns:
            Sorted list with AI scores
        """
        for opp in opportunities:
            # Composite scoring algorithm
            score = 0

            # Probability of profit weight (40%)
            score += opp.get('probability_profit', 0) * 0.4

            # Yield weight (25%)
            normalized_yield = min(opp.get('yield_percent', 0) / 10, 1.0)  # Cap at 10%
            score += normalized_yield * 0.25

            # Recovery percentage weight (20%)
            normalized_recovery = min(opp.get('recovery_percentage', 0) / 100, 1.0)
            score += normalized_recovery * 0.2

            # Strike distance from current price (15%)
            if 'current_price' in opp and 'strike' in opp:
                distance_score = 1 - min(abs(opp['strike'] - opp['current_price']) / opp['current_price'], 1.0)
                score += distance_score * 0.15

            opp['ai_score'] = round(score * 100, 2)

            # Add recommendation text
            opp['recommendation'] = self._generate_recommendation(opp)

        return sorted(opportunities, key=lambda x: x.get('ai_score', 0), reverse=True)

    def _evaluate_recovery_option(self, symbol: str, current_price: float,
                                 strike: float, premium: float,
                                 expiration: str, current_position: Dict) -> Dict:
        """Evaluate a single recovery option"""

        metrics = self.calculate_recovery_metrics(
            current_position, strike, premium, expiration
        )

        metrics['symbol'] = symbol
        metrics['current_price'] = current_price
        metrics['current_strike'] = current_position['current_strike']

        return metrics

    def _get_current_price(self, symbol: str) -> float:
        """Get current stock price"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
        return 0

    def _calculate_support_levels(self, symbol: str, days: int = 50) -> List[float]:
        """Calculate technical support levels"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d")

            if hist.empty:
                return []

            # Find local minima as support levels
            lows = hist['Low'].values
            supports = []

            for i in range(1, len(lows) - 1):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    supports.append(float(lows[i]))

            # Remove duplicates and sort
            supports = sorted(list(set(supports)))

            # Return top 3 most recent support levels
            return supports[-3:] if len(supports) >= 3 else supports

        except Exception as e:
            logger.error(f"Error calculating support levels for {symbol}: {e}")
            return []

    def _calculate_iv_rank(self, symbol: str) -> float:
        """Calculate IV rank (current IV vs 52-week range)"""
        try:
            ticker = yf.Ticker(symbol)

            # Get current IV from nearest ATM option
            exp_dates = ticker.options
            if not exp_dates:
                return 50.0  # Default middle rank

            option_chain = ticker.option_chain(exp_dates[0])
            current_price = self._get_current_price(symbol)

            # Find ATM put
            puts = option_chain.puts
            atm_put = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]

            if not atm_put.empty:
                current_iv = float(atm_put['impliedVolatility'].iloc[0]) * 100

                # For IV rank, we'd need historical IV data
                # Simplified: use current IV as a percentile estimate
                if current_iv < 20:
                    return 25
                elif current_iv < 30:
                    return 50
                elif current_iv < 50:
                    return 75
                else:
                    return 90

        except Exception as e:
            logger.error(f"Error calculating IV rank for {symbol}: {e}")

        return 50.0  # Default middle rank

    def _calculate_volatility(self, symbol: str, days: int = 30) -> float:
        """Calculate historical volatility"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{days}d")

            if len(hist) < 2:
                return 0.3  # Default 30% volatility

            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)

            return float(volatility)

        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return 0.3

    def _calculate_probability_otm(self, current_price: float, strike: float,
                                  volatility: float, days_to_expiry: int) -> float:
        """Calculate probability that option stays out of the money"""
        if days_to_expiry <= 0:
            return 0 if current_price <= strike else 1

        time_to_expiry = days_to_expiry / 365

        # Black-Scholes d2 calculation
        d2 = (np.log(current_price / strike) +
              (self.risk_free_rate - 0.5 * volatility**2) * time_to_expiry) / \
             (volatility * np.sqrt(time_to_expiry))

        # Probability stock stays above strike (put stays OTM)
        prob_otm = norm.cdf(d2)

        return float(prob_otm)

    def _calculate_days_to_expiry(self, expiration_date: str) -> int:
        """Calculate days until expiration"""
        try:
            if isinstance(expiration_date, str):
                exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
            else:
                exp_date = expiration_date

            days = (exp_date - datetime.now()).days
            return max(0, days)
        except Exception as e:
            logger.error(f"Error calculating days to expiry: {e}")
            return 0

    def _generate_recommendation(self, opportunity: Dict) -> str:
        """Generate AI recommendation text"""
        score = opportunity.get('ai_score', 0)
        prob = opportunity.get('probability_profit', 0)
        recovery = opportunity.get('recovery_percentage', 0)

        if score >= 80:
            strength = "STRONG BUY"
            reason = f"Excellent recovery potential ({recovery:.1f}%) with high probability of profit ({prob:.1f}%)"
        elif score >= 60:
            strength = "BUY"
            reason = f"Good balance of premium yield and safety. {recovery:.1f}% recovery potential"
        elif score >= 40:
            strength = "CONSIDER"
            reason = f"Moderate opportunity. Review support levels before entering"
        else:
            strength = "WEAK"
            reason = f"Limited recovery potential. Consider other strikes or wait for better entry"

        return f"{strength}: {reason}"


# Usage example
if __name__ == "__main__":
    analyzer = CSPRecoveryAnalyzer()

    # Example losing position
    sample_position = {
        'symbol': 'AAPL',
        'option_type': 'put',
        'position_type': 'short',
        'strike_price': 180,
        'average_price': 2.50,
        'quantity': -1,
        'expiration_date': '2024-01-19'
    }

    # Analyze position
    losing_positions = analyzer.analyze_losing_positions([sample_position])

    if losing_positions:
        # Find recovery opportunities
        opportunities = analyzer.find_recovery_opportunities(losing_positions[0])

        for opp in opportunities:
            print(f"\nStrike: ${opp['strike']:.2f}")
            print(f"Premium: ${opp['premium']:.2f}")
            print(f"AI Score: {opp['ai_score']:.1f}")
            print(f"Recommendation: {opp['recommendation']}")