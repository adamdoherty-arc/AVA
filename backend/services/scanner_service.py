"""Scanner Service - Wraps the existing PremiumScanner for the API"""

import logging
from typing import List, Dict, Any, Optional
from src.premium_scanner import PremiumScanner
from src.database.query_cache import query_cache

logger = logging.getLogger(__name__)


class ScannerService:
    """Service for scanning option premiums"""

    def __init__(self) -> None:
        self.scanner = PremiumScanner()

    def scan_premiums(
        self,
        symbols: List[str],
        max_price: float = 250,
        min_premium_pct: float = 0.5,
        dte: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Scan for premium opportunities

        Args:
            symbols: List of stock symbols to scan
            max_price: Maximum stock price to consider
            min_premium_pct: Minimum premium as percentage of strike
            dte: Target days to expiration

        Returns:
            List of premium opportunities
        """
        try:
            # Check cache
            cache_key = f"premium_scan_{','.join(sorted(symbols))}_{max_price}_{min_premium_pct}_{dte}"
            cached = query_cache.get(cache_key)
            if cached:
                logger.info("Returning cached premium scan results")
                return cached

            # Run the scan
            results = self.scanner.scan_premiums(
                symbols=symbols,
                max_price=max_price,
                min_premium_pct=min_premium_pct,
                dte=dte
            )

            # Cache for 5 minutes
            query_cache.set(cache_key, results, ttl_seconds=300)

            return results

        except Exception as e:
            logger.error(f"Error scanning premiums: {e}")
            raise

    def scan_multiple_dte(
        self,
        symbols: List[str],
        max_price: float = 250,
        min_premium_pct: float = 0.5,
        dte_targets: List[int] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scan for premiums at multiple DTE targets

        Args:
            symbols: List of stock symbols
            max_price: Maximum stock price
            min_premium_pct: Minimum premium percentage
            dte_targets: List of DTE targets (default: [7, 14, 30, 45])

        Returns:
            Dict mapping DTE to list of opportunities
        """
        if dte_targets is None:
            dte_targets = [7, 14, 30, 45]

        results = {}
        for dte in dte_targets:
            try:
                opportunities = self.scan_premiums(
                    symbols=symbols,
                    max_price=max_price,
                    min_premium_pct=min_premium_pct,
                    dte=dte
                )
                results[str(dte)] = opportunities
            except Exception as e:
                logger.error(f"Error scanning DTE {dte}: {e}")
                results[str(dte)] = []

        return results

    def get_quick_scan(self, dte: int = 30, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Quick scan using predefined watchlist

        Args:
            dte: Target days to expiration
            limit: Maximum number of results

        Returns:
            List of top opportunities
        """
        # Default watchlist for quick scans
        default_symbols = [
            'AAPL', 'AMD', 'AMZN', 'BAC', 'C', 'CCL', 'CSCO', 'F', 'GE',
            'GOOG', 'INTC', 'META', 'MSFT', 'NVDA', 'PLTR', 'PYPL', 'SNAP',
            'SOFI', 'T', 'TSLA', 'UAL', 'UBER', 'WFC', 'XOM'
        ]

        try:
            results = self.scan_premiums(
                symbols=default_symbols,
                max_price=250,
                min_premium_pct=0.5,
                dte=dte
            )
            return results[:limit]
        except Exception as e:
            logger.error(f"Error in quick scan: {e}")
            return []

    def find_assignment_candidates(
        self,
        positions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find CSPs likely to be assigned

        Args:
            positions: Current option positions

        Returns:
            List of assignment candidates
        """
        try:
            return self.scanner.find_assignment_candidates(positions)
        except Exception as e:
            logger.error(f"Error finding assignment candidates: {e}")
            return []


# Singleton instance
_scanner_service = None


def get_scanner_service() -> ScannerService:
    """Get scanner service singleton"""
    global _scanner_service
    if _scanner_service is None:
        _scanner_service = ScannerService()
    return _scanner_service
