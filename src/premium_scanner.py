"""Premium Scanner - Finds the best option premiums for wheel strategy"""

import yfinance as yf
from typing import List, Dict, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

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

    def __init__(self):
        self.min_volume = 100  # Minimum option volume
        self.min_oi = 50  # Minimum open interest
        self.max_workers = 10  # Concurrent threads for scanning

    def _scan_single_symbol(self, symbol: str, max_price: float, min_premium_pct: float, dte: int) -> List[Dict]:
        """Scan a single symbol for premium opportunities."""
        opportunities = []

        try:
            # Get stock info
            ticker = yf.Ticker(symbol)
            info = ticker.info

            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

            # Skip if price too high
            if current_price > max_price or current_price <= 0:
                return []

            # Get options chain
            try:
                # Get available expiration dates
                expirations = ticker.options

                if not expirations:
                    return []

                # Find closest expiration to target DTE
                target_date = datetime.now() + timedelta(days=dte)
                best_expiry = min(expirations,
                                key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_date).days))

                # Get options chain for that date
                opt_chain = ticker.option_chain(best_expiry)
                puts = opt_chain.puts

                if puts.empty:
                    return []

                # Find OTM puts (strike < current price)
                otm_puts = puts[puts['strike'] < current_price * 0.95]  # 5% OTM

                if otm_puts.empty:
                    return []

                # Find best premium
                for _, put in otm_puts.iterrows():
                    strike = put['strike']
                    bid = put['bid']
                    ask = put['ask']
                    volume = put['volume'] or 0
                    oi = put['openInterest'] or 0
                    iv = put['impliedVolatility'] or 0

                    # Skip if no liquidity
                    if volume < self.min_volume and oi < self.min_oi:
                        continue

                    # Use mid price for premium
                    premium = (bid + ask) / 2 if bid > 0 and ask > 0 else bid

                    if premium <= 0:
                        continue

                    # Calculate returns
                    premium_pct = (premium / strike) * 100

                    # Skip if premium too low
                    if premium_pct < min_premium_pct:
                        continue

                    # Calculate annualized return
                    days_to_expiry = (datetime.strptime(best_expiry, '%Y-%m-%d') - datetime.now()).days
                    if days_to_expiry <= 0:
                        continue

                    monthly_return = (premium_pct / days_to_expiry) * 30
                    annual_return = monthly_return * 12

                    opportunities.append({
                        'symbol': symbol,
                        'stock_price': round(current_price, 2),
                        'strike': round(strike, 2),
                        'expiration': best_expiry,
                        'dte': days_to_expiry,
                        'premium': round(premium * 100, 2),  # Premium for 1 contract
                        'premium_pct': premium_pct,
                        'monthly_return': monthly_return,
                        'annual_return': annual_return,
                        'iv': round(iv * 100, 1),
                        'volume': int(volume),
                        'open_interest': int(oi),
                        'bid_ask_spread': round(ask - bid, 3) if ask > 0 and bid > 0 else 0
                    })

            except Exception as e:
                # Skip if options data not available
                pass

        except Exception as e:
            # Skip symbol if error
            pass

        return opportunities

    def scan_premiums(self, symbols: List[str], max_price: float = 50,
                     min_premium_pct: float = 1.0, dte: int = 30) -> List[Dict]:
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