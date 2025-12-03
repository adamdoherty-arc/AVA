"""Premium Scanner - Finds the best option premiums for wheel strategy"""

import yfinance as yf
from typing import List, Dict, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import time
import logging
import random

logger = logging.getLogger(__name__)

# Rate limiting configuration
_last_request_time = 0
_min_request_interval = 0.5  # Minimum seconds between requests

# Per-symbol cache with TTL
_symbol_cache = {}
_cache_ttl = 300  # 5 minutes


def _get_cached_symbol(symbol: str, dte: int) -> Dict | None:
    """Get cached scan result for a symbol."""
    key = f"{symbol}_{dte}"
    if key in _symbol_cache:
        data, expiry = _symbol_cache[key]
        if time.time() < expiry:
            return data
        del _symbol_cache[key]
    return None


def _set_cached_symbol(symbol: str, dte: int, data: List[Dict]):
    """Cache scan result for a symbol."""
    key = f"{symbol}_{dte}"
    _symbol_cache[key] = (data, time.time() + _cache_ttl)


class PremiumScanner:
    """Scans for the best option premiums"""

    def __init__(self) -> None:
        self.min_volume = 10  # Minimum option volume (lowered for better coverage)
        self.min_oi = 10  # Minimum open interest (lowered for better coverage)
        self.max_workers = 5  # Reduced concurrent threads to avoid rate limiting
        self.otm_pct = 0.15  # OTM percentage (15% = strike < 85% of current price)
        self.expiration_range_days = 10  # Scan expirations within +/- this range of target DTE

    def _scan_single_symbol(self, symbol: str, max_price: float, min_premium_pct: float, dte: int) -> List[Dict]:
        """Scan a single symbol for premium opportunities."""
        global _last_request_time
        opportunities = []

        # Rate limiting with jitter to avoid thundering herd
        elapsed = time.time() - _last_request_time
        if elapsed < _min_request_interval:
            sleep_time = _min_request_interval - elapsed + random.uniform(0, 0.2)
            time.sleep(sleep_time)
        _last_request_time = time.time()

        try:
            # Get stock info
            ticker = yf.Ticker(symbol)
            info = ticker.info

            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

            # Skip if price too high
            if current_price > max_price or current_price <= 0:
                logger.debug(f"{symbol}: Skipped - price ${current_price} > max ${max_price} or invalid")
                return []

            # Get options chain
            try:
                # Get available expiration dates
                expirations = ticker.options

                if not expirations:
                    logger.debug(f"{symbol}: No options expirations available")
                    return []

                # Find expirations within range of target DTE
                target_date = datetime.now() + timedelta(days=dte)
                min_date = target_date - timedelta(days=self.expiration_range_days)
                max_date = target_date + timedelta(days=self.expiration_range_days)

                # Get all expirations within the range
                valid_expirations = []
                for exp in expirations:
                    exp_date = datetime.strptime(exp, '%Y-%m-%d')
                    if min_date <= exp_date <= max_date:
                        valid_expirations.append(exp)

                # If no expirations in range, use the closest one
                if not valid_expirations:
                    best_expiry = min(expirations,
                                    key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_date).days))
                    valid_expirations = [best_expiry]

                logger.debug(f"{symbol}: Found {len(valid_expirations)} valid expirations near {dte} DTE")

                # Scan each valid expiration
                for expiry in valid_expirations:
                    try:
                        opt_chain = ticker.option_chain(expiry)
                        puts = opt_chain.puts

                        if puts.empty:
                            continue

                        # Find OTM puts - using configurable percentage (default 15%)
                        # Strike < current_price * (1 - otm_pct) means deeper OTM
                        # We want strikes between (current_price * (1-otm_pct)) and current_price
                        min_strike = current_price * (1 - self.otm_pct)
                        otm_puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] < current_price)]

                        if otm_puts.empty:
                            # If no OTM puts in range, try slightly ITM puts too (within 3%)
                            otm_puts = puts[(puts['strike'] >= min_strike) & (puts['strike'] <= current_price * 1.03)]

                        if otm_puts.empty:
                            logger.debug(f"{symbol} {expiry}: No puts in range ${min_strike:.2f} - ${current_price:.2f}")
                            continue

                        # Find best premiums
                        for _, put in otm_puts.iterrows():
                            strike = put['strike']
                            bid = put['bid']
                            ask = put['ask']
                            volume = put['volume'] or 0
                            oi = put['openInterest'] or 0
                            iv = put['impliedVolatility'] or 0

                            # Skip if no liquidity - require EITHER volume OR open interest
                            if volume < self.min_volume and oi < self.min_oi:
                                continue

                            # Use mid price for premium, fallback to lastPrice if market closed
                            if bid > 0 and ask > 0:
                                premium = (bid + ask) / 2
                            elif bid > 0:
                                premium = bid
                            else:
                                # Market may be closed - use lastPrice as fallback
                                last_price = put.get('lastPrice', 0) or 0
                                premium = last_price

                            if premium <= 0:
                                continue

                            # Calculate returns
                            premium_pct = (premium / strike) * 100

                            # Skip if premium too low
                            if premium_pct < min_premium_pct:
                                continue

                            # Calculate annualized return
                            days_to_expiry = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days
                            if days_to_expiry <= 0:
                                continue

                            monthly_return = (premium_pct / days_to_expiry) * 30
                            annual_return = monthly_return * 12

                            # Calculate delta approximation if not provided
                            delta = put.get('delta', 0)
                            if delta == 0:
                                # Rough delta approximation based on moneyness
                                moneyness = strike / current_price
                                delta = -0.5 * moneyness if moneyness < 1 else -0.5

                            # Calculate additional quality metrics
                            bid_ask_spread = ask - bid if ask > 0 and bid > 0 else 0
                            spread_pct = (bid_ask_spread / premium * 100) if premium > 0 else 0

                            # Spread quality rating
                            if spread_pct <= 10:
                                spread_quality = 'tight'
                            elif spread_pct <= 25:
                                spread_quality = 'moderate'
                            else:
                                spread_quality = 'wide'

                            # Liquidity score (0-100 based on volume and OI)
                            vol_score = min(50, (volume / 100) * 10) if volume > 0 else 0
                            oi_score = min(50, (oi / 500) * 10) if oi > 0 else 0
                            liquidity_score = int(vol_score + oi_score)

                            # Moneyness percentage (negative = OTM, positive = ITM)
                            otm_pct = round((current_price - strike) / current_price * 100, 2)

                            # Theta estimate (rough approximation)
                            # Theta ~ -premium / DTE for ATM options, adjusted for OTM
                            theta_estimate = round(-premium / days_to_expiry * (1 - abs(otm_pct) / 100), 4)

                            # Collateral required for 1 CSP contract
                            collateral = strike * 100

                            opportunities.append({
                                'symbol': symbol,
                                'stock_price': round(current_price, 2),
                                'strike': round(strike, 2),
                                'expiration': expiry,
                                'dte': days_to_expiry,
                                'bid': round(bid, 2) if bid else 0,
                                'ask': round(ask, 2) if ask else 0,
                                'premium': round(premium * 100, 2),  # Premium for 1 contract
                                'premium_pct': round(premium_pct, 2),
                                'monthly_return': round(monthly_return, 2),
                                'annual_return': round(annual_return, 2),
                                'iv': round(iv * 100, 1),
                                'volume': int(volume),
                                'open_interest': int(oi),
                                'bid_ask_spread': round(bid_ask_spread, 3),
                                'spread_pct': round(spread_pct, 1),
                                'spread_quality': spread_quality,
                                'liquidity_score': liquidity_score,
                                'delta': round(delta, 3) if isinstance(delta, (int, float)) else 0,
                                'theta': theta_estimate,
                                'otm_pct': otm_pct,
                                'collateral': round(collateral, 2)
                            })

                    except Exception as e:
                        logger.debug(f"{symbol} {expiry}: Error scanning - {str(e)[:100]}")
                        continue

                logger.debug(f"{symbol}: Found {len(opportunities)} opportunities")

            except Exception as e:
                logger.warning(f"{symbol}: Options chain error - {str(e)[:100]}")

        except Exception as e:
            logger.warning(f"{symbol}: Stock info error - {str(e)[:100]}")

        return opportunities

    def scan_premiums(self, symbols: List[str], max_price: float = 250,
                     min_premium_pct: float = 0.5, dte: int = 30) -> List[Dict]:
        """
        Scan symbols for best put premiums using concurrent execution.

        Args:
            symbols: List of stock symbols to scan
            max_price: Maximum stock price
            min_premium_pct: Minimum premium as % of strike
            dte: Target days to expiration

        Returns:
            List of premium opportunities sorted by return
        """
        opportunities = []
        symbols_to_scan = []

        # Check cache first for each symbol
        for symbol in symbols:
            cached = _get_cached_symbol(symbol, dte)
            if cached is not None:
                # Filter cached results by current params
                for opp in cached:
                    if opp.get('stock_price', 999) <= max_price and opp.get('premium_pct', 0) >= min_premium_pct:
                        opportunities.append(opp)
                logger.debug(f"Cache hit for {symbol}")
            else:
                symbols_to_scan.append(symbol)

        if symbols_to_scan:
            logger.info(f"Scanning {len(symbols_to_scan)} symbols concurrently (cached: {len(symbols) - len(symbols_to_scan)})")

            # Use ThreadPoolExecutor for concurrent scanning
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all scan tasks
                future_to_symbol = {
                    executor.submit(self._scan_single_symbol, symbol, max_price, min_premium_pct, dte): symbol
                    for symbol in symbols_to_scan
                }

                # Collect results as they complete
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        symbol_results = future.result()
                        # Cache results for this symbol
                        _set_cached_symbol(symbol, dte, symbol_results)
                        opportunities.extend(symbol_results)
                    except Exception as e:
                        logger.warning(f"Error scanning {symbol}: {e}")

        # Sort by monthly return (best first)
        opportunities.sort(key=lambda x: x['monthly_return'], reverse=True)

        return opportunities

    def scan_all_stocks_under(self, max_price: float) -> List[str]:
        """
        Get a list of liquid stocks under a certain price

        Args:
            max_price: Maximum stock price

        Returns:
            List of stock symbols
        """

        # No default stocks - return empty list
        stocks = []

        # Filter by current price
        valid_stocks = []
        for symbol in stocks:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

                if 0 < price <= max_price:
                    valid_stocks.append(symbol)
            except:
                continue

        return valid_stocks

    def find_assignment_candidates(self, positions: List[Dict]) -> List[Dict]:
        """
        Find CSPs that are likely to be assigned (for selling CCs)

        Args:
            positions: Current CSP positions

        Returns:
            List of positions likely to be assigned
        """

        candidates = []

        for pos in positions:
            if pos.get('Type') != 'CSP':
                continue

            symbol = pos['Symbol']
            strike = pos.get('Strike', 0)

            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                current_price = info.get('currentPrice', 0)

                # Check if ITM
                if current_price < strike:
                    pct_itm = ((strike - current_price) / strike) * 100

                    candidates.append({
                        'symbol': symbol,
                        'strike': strike,
                        'current_price': current_price,
                        'pct_itm': pct_itm,
                        'days_to_expiry': pos.get('Days to Expiry', 0),
                        'assignment_probability': min(90, 50 + pct_itm * 10)  # Simple probability estimate
                    })
            except:
                continue

        return candidates