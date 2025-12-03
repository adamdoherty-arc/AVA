"""Enhanced Options Fetcher with Multiple Expirations and Greeks
Fetches options for 7, 14, 21, 30, 45 DTE with delta calculations
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import robin_stocks.robinhood as rh
import numpy as np
from scipy.stats import norm
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedOptionsFetcher:
    """Fetches options data with multiple expirations and Greeks"""

    def __init__(self) -> None:
        self.logged_in = False
        self.risk_free_rate = 0.045  # Current risk-free rate (~4.5%)

    def login_robinhood(self) -> None:
        """Login to Robinhood once"""
        if self.logged_in:
            return True

        try:
            username = os.getenv('ROBINHOOD_USERNAME')
            password = os.getenv('ROBINHOOD_PASSWORD')

            if not username or not password:
                return False

            rh.authentication.login(
                username=username,
                password=password,
                expiresIn=86400,
                store_session=True
            )
            self.logged_in = True
            logger.info("Logged into Robinhood")
            return True

        except Exception as e:
            logger.error(f"Robinhood login failed: {e}")
            return False

    def calculate_delta(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'put') -> float:
        """
        Calculate option delta using Black-Scholes

        S: Current stock price
        K: Strike price
        T: Time to expiration (in years)
        r: Risk-free rate
        sigma: Implied volatility
        option_type: 'call' or 'put'
        """
        try:
            if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
                return 0.0

            d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

            if option_type == 'put':
                delta = norm.cdf(d1) - 1
            else:
                delta = norm.cdf(d1)

            return round(delta, 4)

        except Exception as e:
            logger.debug(f"Delta calculation error: {e}")
            return 0.0

    def get_all_expirations_data(self, symbol: str, target_dtes: List[int] = [7, 14, 21, 30, 45]) -> List[Dict]:
        """
        Get options data for multiple expirations

        Returns list of dicts with all expiration data
        """
        if not self.login_robinhood():
            return []

        try:
            # Get current price
            quote = rh.stocks.get_stock_quote_by_symbol(symbol)
            if not quote:
                return []

            current_price = float(quote.get('last_trade_price', 0))
            if not current_price:
                return []

            # Get all available expirations
            chains = rh.options.get_chains(symbol)
            if not chains:
                return []

            all_exp_dates = chains.get('expiration_dates', [])
            if not all_exp_dates:
                return []

            results = []

            # For each target DTE, find closest expiration
            for target_dte in target_dtes:
                target_date = datetime.now() + timedelta(days=target_dte)

                # Find closest expiration to target
                closest_exp = min(
                    all_exp_dates,
                    key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_date).days)
                )

                exp_date = datetime.strptime(closest_exp, '%Y-%m-%d')
                actual_dte = (exp_date - datetime.now()).days

                # Skip if too far from target (>5 days difference for weekly, >10 for monthly)
                max_diff = 5 if target_dte <= 14 else 10
                if abs(actual_dte - target_dte) > max_diff:
                    continue

                # Get put options for this expiration
                puts = rh.options.find_options_by_expiration(
                    symbol,
                    closest_exp,
                    optionType='put'
                )

                if not puts:
                    continue

                # Find strike with delta closest to -0.30 (30 delta)
                # First, calculate delta for each put and find closest to -0.30
                # IMPORTANT: For CSPs, strike MUST be below current stock price (OTM)
                target_delta = -0.30
                best_put = None
                best_delta_diff = float('inf')

                for put in puts:
                    strike = float(put.get('strike_price', 0))
                    if strike <= 0:
                        continue

                    # CRITICAL FIX: Only consider strikes BELOW stock price for CSPs
                    # CSPs should be out-of-the-money (OTM), meaning strike < stock price
                    if strike >= current_price:
                        continue  # Skip ITM and ATM strikes

                    # Calculate time to expiration in years
                    T = actual_dte / 365.0

                    # Use IV from market if available, otherwise estimate
                    # We'll do a quick estimate first, then refine after getting market data
                    estimated_iv = 0.35  # Default estimate

                    # Calculate delta
                    delta = self.calculate_delta(
                        S=current_price,
                        K=strike,
                        T=T,
                        r=0.045,  # Risk-free rate
                        sigma=estimated_iv,
                        option_type='put'
                    )

                    delta_diff = abs(delta - target_delta)
                    if delta_diff < best_delta_diff:
                        best_delta_diff = delta_diff
                        best_put = put

                if not best_put:
                    continue

                closest_put = best_put

                # Get market data
                option_id = closest_put['id']
                market_data = rh.options.get_option_market_data_by_id(option_id)

                if not market_data:
                    continue

                data = market_data[0]
                strike_price = float(closest_put.get('strike_price', 0))
                bid = float(data.get('bid_price', 0))
                ask = float(data.get('ask_price', 0))
                mid = (bid + ask) / 2
                premium = mid * 100
                volume = int(data.get('volume', 0))
                oi = int(data.get('open_interest', 0))
                iv = float(data.get('implied_volatility', 0))

                # Calculate returns
                capital = strike_price * 100
                premium_pct = (premium / capital * 100) if capital > 0 else 0
                T_years = actual_dte / 365.0
                monthly_return = (premium_pct / actual_dte * 30) if actual_dte > 0 else 0
                annual_return = (premium_pct / actual_dte * 365) if actual_dte > 0 else 0

                # Calculate delta
                delta = self.calculate_delta(
                    S=current_price,
                    K=strike_price,
                    T=T_years,
                    r=self.risk_free_rate,
                    sigma=iv,
                    option_type='put'
                )

                # Calculate probability of profit (based on delta)
                # For puts, probability of profit ≈ 1 + delta (since delta is negative)
                prob_profit = round((1 + delta) * 100, 1)

                # Calculate break-even
                breakeven = strike_price - (premium / 100)

                # Downside protection
                downside_protection = ((current_price - breakeven) / current_price * 100)

                results.append({
                    'target_dte': target_dte,
                    'actual_dte': actual_dte,
                    'expiration_date': closest_exp,
                    'strike_price': strike_price,
                    'current_price': current_price,
                    'bid': bid,
                    'ask': ask,
                    'mid': mid,
                    'premium': premium,
                    'premium_pct': premium_pct,
                    'monthly_return': monthly_return,
                    'annual_return': annual_return,
                    'delta': delta,
                    'prob_profit': prob_profit,
                    'iv': iv * 100,
                    'volume': volume,
                    'open_interest': oi,
                    'breakeven': breakeven,
                    'downside_protection': downside_protection,
                    'strike_to_spot': (strike_price / current_price - 1) * 100  # % OTM
                })

            return results

        except Exception as e:
            logger.error(f"Error fetching options for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_best_opportunities(self, symbols: List[str], min_monthly_return: float = 1.0) -> List[Dict]:
        """
        Scan multiple symbols and return best opportunities

        Returns list sorted by monthly return
        """
        all_opportunities = []

        for symbol in symbols:
            logger.info(f"Scanning {symbol}...")
            options_data = self.get_all_expirations_data(symbol)

            for opt in options_data:
                if opt['monthly_return'] >= min_monthly_return:
                    opt['symbol'] = symbol
                    all_opportunities.append(opt)

        # Sort by monthly return (descending)
        all_opportunities.sort(key=lambda x: x['monthly_return'], reverse=True)

        return all_opportunities


if __name__ == "__main__":
    # Test the enhanced fetcher
    from dotenv import load_dotenv
    load_dotenv()

    fetcher = EnhancedOptionsFetcher()

    test_symbols = ['AAPL', 'AMD', 'COIN']

    print("\n" + "="*80)
    print("ENHANCED OPTIONS SCANNER - Multiple Expirations with Delta")
    print("="*80)

    for symbol in test_symbols:
        print(f"\n{'='*80}")
        print(f"{symbol} - Cash-Secured Put Opportunities")
        print(f"{'='*80}")

        options = fetcher.get_all_expirations_data(symbol)

        if options:
            print(f"\nFound {len(options)} expiration dates\n")

            # Header
            print(f"{'DTE':<6} {'Exp Date':<12} {'Strike':<8} {'Premium':<10} {'Monthly%':<10} {'Delta':<8} {'Prob%':<8} {'IV%':<6}")
            print("-" * 80)

            for opt in options:
                print(f"{opt['actual_dte']:<6} "
                      f"{opt['expiration_date']:<12} "
                      f"${opt['strike_price']:<7.2f} "
                      f"${opt['premium']:<9.2f} "
                      f"{opt['monthly_return']:<9.2f}% "
                      f"{opt['delta']:<7.3f} "
                      f"{opt['prob_profit']:<7.1f}% "
                      f"{opt['iv']:<5.1f}%")

        else:
            print(f"No options data available for {symbol}")

    print("\n" + "="*80)
    print("Scan Complete!")
    print("="*80)
