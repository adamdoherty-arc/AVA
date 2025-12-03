"""TradingView Watchlist Integration - Pull symbols from TradingView watchlists"""

import os
import json
import requests
from typing import List, Dict, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import yfinance as yf
import time

class TradingViewWatchlist:
    """Pull watchlists from TradingView"""
    
    def __init__(self) -> None:
        self.username = os.getenv('TRADINGVIEW_USERNAME')
        self.password = os.getenv('TRADINGVIEW_PASSWORD')
        # Fix: Use correct environment variable or default to 50
        self.max_price = float(os.getenv('MAX_STOCK_PRICE', '50'))
        
    def get_watchlist_symbols_simple(self) -> List[str]:
        """
        Get symbols from predefined watchlists for wheel strategy.
        Returns popular stocks under $50 suitable for wheel strategy.
        """
        
        # No default symbols - return empty if no watchlist loaded
        watchlist_symbols = []
        
        # Filter by current price
        filtered_symbols = []
        
        for symbol in watchlist_symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                
                # Only include if under max price and has options
                if 0 < price <= self.max_price:
                    # Check if has options
                    options = ticker.options
                    if options and len(options) > 0:
                        filtered_symbols.append(symbol)
                        
            except Exception:
                continue
                
        return filtered_symbols
    
    def get_nvda_watchlist(self) -> List[str]:
        """Get NVDA and related semiconductor/AI stocks - empty by default"""
        return []  # No hardcoded stocks

    def get_comprehensive_premiums(self, symbols: Optional[List[str]] = None,
                                  max_price: Optional[float] = None) -> Dict:
        """
        Get comprehensive premium analysis for multiple expiration dates.

        Args:
            symbols: List of symbols to analyze (defaults to NVDA watchlist)
            max_price: Maximum stock price to consider (None = no limit)

        Returns:
            Dictionary with detailed premium analysis for each symbol
        """

        if not symbols:
            return {}  # Return empty if no symbols provided

        from src.premium_scanner import PremiumScanner
        from datetime import datetime, timedelta

        scanner = PremiumScanner()
        comprehensive_data = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

                # Skip if price limit is set and exceeded
                if max_price and current_price > max_price:
                    continue

                symbol_data = {
                    'symbol': symbol,
                    'current_price': current_price,
                    'expirations': []
                }

                # Get all available expiration dates
                expirations = ticker.options[:5]  # Get next 5 expiration dates

                for expiry in expirations:
                    try:
                        # Calculate days to expiry
                        exp_date = datetime.strptime(expiry, '%Y-%m-%d')
                        days_to_expiry = (exp_date - datetime.now()).days

                        # Skip if > 45 days
                        if days_to_expiry > 45 or days_to_expiry < 7:
                            continue

                        # Get option chain
                        opt_chain = ticker.option_chain(expiry)
                        puts = opt_chain.puts

                        if puts.empty:
                            continue

                        # Find best OTM puts (5-10% OTM)
                        otm_puts = puts[
                            (puts['strike'] <= current_price * 0.95) &
                            (puts['strike'] >= current_price * 0.90)
                        ]

                        if otm_puts.empty:
                            continue

                        # Get best premium for this expiration
                        best_put = None
                        best_return = 0

                        for _, put in otm_puts.iterrows():
                            strike = put['strike']
                            bid = put['bid']
                            ask = put['ask']
                            volume = put['volume'] or 0
                            oi = put['openInterest'] or 0
                            iv = put['impliedVolatility'] or 0

                            # Skip if no liquidity
                            if volume < 10 and oi < 50:
                                continue

                            # Use mid price for premium
                            premium = (bid + ask) / 2 if bid > 0 and ask > 0 else bid

                            if premium <= 0:
                                continue

                            # Calculate returns
                            capital_required = strike * 100  # Per contract
                            premium_total = premium * 100
                            return_pct = (premium_total / capital_required) * 100
                            monthly_return = (return_pct / days_to_expiry) * 30 if days_to_expiry > 0 else 0
                            annual_return = monthly_return * 12

                            if return_pct > best_return:
                                best_return = return_pct
                                best_put = {
                                    'strike': strike,
                                    'premium_per_share': premium,
                                    'premium_total': premium_total,
                                    'capital_required': capital_required,
                                    'return_pct': return_pct,
                                    'monthly_return': monthly_return,
                                    'annual_return': annual_return,
                                    'iv': iv * 100,
                                    'volume': volume,
                                    'open_interest': oi,
                                    'bid_ask_spread': ask - bid if ask > 0 and bid > 0 else 0,
                                    'days_to_expiry': days_to_expiry,
                                    'expiration': expiry
                                }

                        if best_put:
                            symbol_data['expirations'].append(best_put)

                    except Exception:
                        continue

                if symbol_data['expirations']:
                    comprehensive_data[symbol] = symbol_data

            except Exception:
                continue

        return comprehensive_data

    def get_best_premiums(self, symbols: Optional[List[str]] = None) -> List[Dict]:
        """
        Find best option premiums from watchlist symbols.

        Returns:
            List of best premium opportunities sorted by return
        """

        if not symbols:
            symbols = self.get_watchlist_symbols_simple()

        from src.premium_scanner import PremiumScanner

        scanner = PremiumScanner()

        # Scan for best premiums
        opportunities = scanner.scan_premiums(
            symbols=symbols,
            max_price=self.max_price if self.max_price > 1 else 50,  # Fix if max_price is too low
            min_premium_pct=1.0,  # At least 1% premium
            dte=30  # Target 30 days to expiration
        )

        # Sort by monthly return
        opportunities.sort(key=lambda x: x.get('monthly_return', 0), reverse=True)

        # Return top 20 opportunities
        return opportunities[:20]
    
    def get_watchlist_analysis(self) -> Dict:
        """
        Get complete analysis of watchlist for wheel strategy.
        """
        
        symbols = self.get_watchlist_symbols_simple()
        
        analysis = {
            'total_symbols': len(symbols),
            'symbols': symbols,
            'best_premiums': [],
            'by_sector': {},
            'high_iv': [],
            'best_weekly': [],
            'best_monthly': []
        }
        
        # Get best overall premiums
        premiums = self.get_best_premiums(symbols)
        analysis['best_premiums'] = premiums[:10]
        
        # Find high IV plays
        high_iv = [p for p in premiums if p.get('iv', 0) > 40]
        analysis['high_iv'] = high_iv[:5]
        
        # Best weekly (7-14 DTE)
        weekly = [p for p in premiums if 7 <= p.get('dte', 0) <= 14]
        weekly.sort(key=lambda x: x.get('annual_return', 0), reverse=True)
        analysis['best_weekly'] = weekly[:5]
        
        # Best monthly (25-35 DTE)  
        monthly = [p for p in premiums if 25 <= p.get('dte', 0) <= 35]
        monthly.sort(key=lambda x: x.get('monthly_return', 0), reverse=True)
        analysis['best_monthly'] = monthly[:5]
        
        return analysis