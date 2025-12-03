"""
Calendar Spread Analyzer - AI-powered calendar spread opportunity finder

Analyzes stocks for optimal calendar spread setups based on:
- Time decay potential (theta differential)
- Implied volatility (low IV < 30% preferred)
- Price stability (within 1-2 standard deviations)
- Optimal expiration timing (30-45 DTE for short leg)
"""

import robin_stocks.robinhood as rh
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import math
import logging

from src.services.rate_limiter import rate_limit

logger = logging.getLogger(__name__)


# PERFORMANCE FIX: Rate-limited Robinhood API wrappers to prevent account bans
# Robinhood allows ~60 calls/minute. Without rate limiting, nested loops can hit 200+ calls/minute.

@rate_limit("robinhood", tokens=1, timeout=30)
def get_chains_rate_limited(symbol: str):
    """Rate-limited wrapper for rh.get_chains()"""
    return rh.get_chains(symbol)


@rate_limit("robinhood", tokens=1, timeout=30)
def get_chain_expirations_rate_limited(chain_id: str):
    """Rate-limited wrapper for rh.get_chain_expirations()"""
    return rh.get_chain_expirations(chain_id)


@rate_limit("robinhood", tokens=1, timeout=30)
def get_chain_strikes_rate_limited(chain_id: str):
    """Rate-limited wrapper for rh.get_chain_strikes()"""
    return rh.get_chain_strikes(chain_id)


@rate_limit("robinhood", tokens=1, timeout=30)
def find_options_rate_limited(symbol: str, expiration: str, strike: str, option_type: str):
    """Rate-limited wrapper for rh.find_options_by_expiration_and_strike()"""
    return rh.find_options_by_expiration_and_strike(symbol, expiration, strike, option_type)


@rate_limit("robinhood", tokens=1, timeout=30)
def get_option_market_data_rate_limited(option_id: str):
    """Rate-limited wrapper for rh.get_option_market_data_by_id()"""
    return rh.get_option_market_data_by_id(option_id)


class CalendarSpreadAnalyzer:
    """Analyzes and scores calendar spread opportunities"""

    def __init__(self) -> None:
        self.min_iv = 0.10  # Minimum IV to consider (10%)
        self.max_iv = 0.30  # Maximum IV for ideal spreads (30%)
        self.target_short_dte = (30, 45)  # Target DTE range for short leg
        self.target_long_dte = (60, 90)  # Target DTE range for long leg
        self.min_liquidity_volume = 50  # Minimum volume
        self.min_open_interest = 100  # Minimum OI

    def analyze_symbol(self, symbol: str, stock_price: float) -> List[Dict]:
        """
        Analyze a symbol for calendar spread opportunities

        Args:
            symbol: Stock ticker
            stock_price: Current stock price

        Returns:
            List of calendar spread opportunities with scores
        """
        try:
            opportunities = []

            # Get option chains (rate-limited)
            chains = get_chains_rate_limited(symbol)
            if not chains:
                return []

            chain_id = chains[0]['id']

            # Get expiration dates (rate-limited)
            expirations = get_chain_expirations_rate_limited(chain_id)
            if len(expirations) < 2:
                return []  # Need at least 2 expirations

            # Calculate DTE for each expiration
            exp_with_dte = []
            for exp_date in expirations:
                exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                dte = (exp_dt - datetime.now()).days
                exp_with_dte.append((exp_date, dte))

            # Find suitable short and long leg expirations
            short_exps = [e for e in exp_with_dte if self.target_short_dte[0] <= e[1] <= self.target_short_dte[1]]
            long_exps = [e for e in exp_with_dte if self.target_long_dte[0] <= e[1] <= self.target_long_dte[1]]

            if not short_exps or not long_exps:
                return []

            # Analyze ATM and near-ATM strikes
            strikes_to_analyze = self._get_strikes_near_price(symbol, stock_price, chain_id)

            for strike in strikes_to_analyze:
                for short_exp, short_dte in short_exps:
                    for long_exp, long_dte in long_exps:
                        # Analyze both call and put calendars
                        for option_type in ['call', 'put']:
                            spread = self._analyze_spread(
                                symbol, stock_price, strike,
                                short_exp, short_dte,
                                long_exp, long_dte,
                                option_type
                            )

                            if spread and spread['score'] >= 50:  # Minimum score threshold
                                opportunities.append(spread)

            # Sort by score (highest first)
            opportunities.sort(key=lambda x: x['score'], reverse=True)

            return opportunities[:10]  # Return top 10

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return []

    def _get_strikes_near_price(self, symbol: str, stock_price: float, chain_id: str) -> List[float]:
        """Get strikes within 10% of stock price"""
        try:
            # Get option strikes (rate-limited)
            strikes_data = get_chain_strikes_rate_limited(chain_id)
            all_strikes = [float(s) for s in strikes_data]

            # Filter to strikes within 10% of stock price
            lower_bound = stock_price * 0.90
            upper_bound = stock_price * 1.10

            near_strikes = [s for s in all_strikes if lower_bound <= s <= upper_bound]

            # Return top 5 strikes closest to ATM
            near_strikes.sort(key=lambda x: abs(x - stock_price))
            return near_strikes[:5]

        except Exception as e:
            logger.error(f"Error getting strikes for {symbol}: {e}")
            return []

    def _analyze_spread(
        self,
        symbol: str,
        stock_price: float,
        strike: float,
        short_exp: str,
        short_dte: int,
        long_exp: str,
        long_dte: int,
        option_type: str
    ) -> Optional[Dict]:
        """Analyze a specific calendar spread setup"""
        try:
            # Get option data for both legs
            short_option = self._get_option_data(symbol, short_exp, strike, option_type)
            long_option = self._get_option_data(symbol, long_exp, strike, option_type)

            if not short_option or not long_option:
                return None

            # Check liquidity
            if short_option['volume'] < self.min_liquidity_volume or \
               long_option['volume'] < self.min_liquidity_volume:
                return None

            if short_option['open_interest'] < self.min_open_interest or \
               long_option['open_interest'] < self.min_open_interest:
                return None

            # Calculate spread cost (net debit)
            short_premium = (float(short_option['bid']) + float(short_option['ask'])) / 2 * 100
            long_premium = (float(long_option['bid']) + float(long_option['ask'])) / 2 * 100
            net_debit = long_premium - short_premium

            if net_debit <= 0:  # Invalid spread
                return None

            # Calculate max loss and estimated max profit
            max_loss = net_debit
            max_profit_estimate = net_debit * 0.50  # Conservative estimate: 50% return

            # Calculate IV metrics
            short_iv = float(short_option.get('implied_volatility', 0))
            long_iv = float(long_option.get('implied_volatility', 0))
            avg_iv = (short_iv + long_iv) / 2

            # Calculate Greeks
            short_theta = abs(float(short_option.get('theta', 0)))
            long_theta = abs(float(long_option.get('theta', 0)))
            theta_differential = short_theta - long_theta  # Positive = good for calendar

            short_delta = abs(float(short_option.get('delta', 0)))

            # Calculate score using AI-based weighting
            score = self._calculate_ai_score(
                stock_price=stock_price,
                strike=strike,
                avg_iv=avg_iv,
                theta_differential=theta_differential,
                short_delta=short_delta,
                short_dte=short_dte,
                net_debit=net_debit,
                liquidity_score=(short_option['volume'] + long_option['volume']) / 2
            )

            # Build spread dictionary
            spread = {
                'symbol': symbol,
                'type': f"{option_type.capitalize()} Calendar",
                'strike': strike,
                'stock_price': stock_price,
                'short_exp': short_exp,
                'short_dte': short_dte,
                'long_exp': long_exp,
                'long_dte': long_dte,
                'short_premium': short_premium,
                'long_premium': long_premium,
                'net_debit': net_debit,
                'max_loss': max_loss,
                'max_profit_estimate': max_profit_estimate,
                'profit_potential': (max_profit_estimate / max_loss) * 100 if max_loss > 0 else 0,
                'avg_iv': avg_iv * 100,  # Convert to percentage
                'theta_differential': theta_differential,
                'short_delta': short_delta,
                'short_volume': short_option['volume'],
                'long_volume': long_option['volume'],
                'short_oi': short_option['open_interest'],
                'long_oi': long_option['open_interest'],
                'score': score,
                'recommendation': self._get_recommendation(score, avg_iv)
            }

            return spread

        except Exception as e:
            logger.error(f"Error analyzing spread for {symbol} {strike} {option_type}: {e}")
            return None

    def _get_option_data(self, symbol: str, expiration: str, strike: float, option_type: str) -> Optional[Dict]:
        """Get option data from Robinhood"""
        try:
            # Find options (rate-limited)
            options = find_options_rate_limited(
                symbol,
                expiration,
                str(strike),
                option_type
            )

            if not options or len(options) == 0:
                return None

            option_data = options[0]

            # Get market data (rate-limited)
            market_data = get_option_market_data_rate_limited(option_data['id'])
            if not market_data or len(market_data) == 0:
                return None

            market = market_data[0]

            return {
                'bid': market.get('bid_price', 0),
                'ask': market.get('ask_price', 0),
                'volume': int(market.get('volume', 0)),
                'open_interest': int(market.get('open_interest', 0)),
                'implied_volatility': market.get('implied_volatility', 0),
                'delta': market.get('delta', 0),
                'theta': market.get('theta', 0),
                'gamma': market.get('gamma', 0),
                'vega': market.get('vega', 0)
            }

        except Exception as e:
            logger.error(f"Error getting option data: {e}")
            return None

    def _calculate_ai_score(
        self,
        stock_price: float,
        strike: float,
        avg_iv: float,
        theta_differential: float,
        short_delta: float,
        short_dte: int,
        net_debit: float,
        liquidity_score: float
    ) -> float:
        """
        Calculate AI-powered score (0-100) for calendar spread

        Weighting factors:
        - Theta differential (30%): Higher is better
        - IV level (25%): Lower is better (< 30% ideal)
        - Moneyness (20%): ATM is best
        - Timing (15%): 30-45 DTE is ideal
        - Liquidity (10%): Higher is better
        """
        score = 0.0

        # 1. Theta Differential Score (30 points)
        # Normalize theta diff (typically 0-0.10 range)
        theta_score = min(theta_differential / 0.10, 1.0) * 30
        score += theta_score

        # 2. IV Score (25 points)
        # Best: IV < 30%, Good: 30-40%, Poor: > 40%
        if avg_iv <= 0.30:
            iv_score = 25
        elif avg_iv <= 0.40:
            iv_score = 15
        elif avg_iv <= 0.50:
            iv_score = 8
        else:
            iv_score = 2
        score += iv_score

        # 3. Moneyness Score (20 points)
        # Best: ATM (within 2%), Good: within 5%
        price_diff_pct = abs(strike - stock_price) / stock_price
        if price_diff_pct <= 0.02:  # Within 2%
            moneyness_score = 20
        elif price_diff_pct <= 0.05:  # Within 5%
            moneyness_score = 12
        elif price_diff_pct <= 0.10:  # Within 10%
            moneyness_score = 5
        else:
            moneyness_score = 0
        score += moneyness_score

        # 4. Timing Score (15 points)
        # Ideal: 30-45 DTE
        if 30 <= short_dte <= 45:
            timing_score = 15
        elif 21 <= short_dte <= 60:
            timing_score = 10
        elif 14 <= short_dte <= 70:
            timing_score = 5
        else:
            timing_score = 0
        score += timing_score

        # 5. Liquidity Score (10 points)
        # Normalize volume (100+ is good)
        liquidity_normalized = min(liquidity_score / 100, 1.0) * 10
        score += liquidity_normalized

        return round(score, 1)

    def _get_recommendation(self, score: float, avg_iv: float) -> str:
        """Get recommendation based on score and IV"""
        if score >= 80:
            return "⭐ Excellent - Strong Setup"
        elif score >= 70:
            return "✅ Good - Consider Entry"
        elif score >= 60:
            return "👍 Fair - Watch Closely"
        elif score >= 50:
            return "⚠️ Marginal - Proceed with Caution"
        else:
            return "❌ Poor - Avoid"

    def calculate_max_profit_loss(
        self,
        net_debit: float,
        long_option_value_at_short_exp: float
    ) -> Tuple[float, float]:
        """
        Calculate max profit and loss for calendar spread

        Args:
            net_debit: Initial cost of spread
            long_option_value_at_short_exp: Estimated value of long option when short expires

        Returns:
            (max_profit, max_loss)
        """
        max_loss = net_debit
        max_profit = long_option_value_at_short_exp - net_debit

        return (max(max_profit, 0), max_loss)
