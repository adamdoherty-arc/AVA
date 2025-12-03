"""
Scanner Service V2 - Enhanced scanner with universe integration

Provides:
- Premium scanning with universe pre-filtering
- Multi-DTE comparison
- Sector/category-based scanning
- Progress tracking for streaming
- Robust caching
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import threading

from src.premium_scanner import PremiumScanner  # Keep original for fallback
from src.database.query_cache import query_cache

logger = logging.getLogger(__name__)


# Lazy imports to avoid circular dependencies
def _get_universe_service():
    try:
        from backend.services.universe_service import get_universe_service, UniverseFilter, AssetType
        return get_universe_service(), UniverseFilter, AssetType
    except ImportError:
        logger.warning("UniverseService not available")
        return None, None, None


def _get_scanner_v2():
    try:
        from src.premium_scanner_v2 import get_premium_scanner_v2, ScanProgress
        return get_premium_scanner_v2(), ScanProgress
    except ImportError:
        logger.warning("PremiumScannerV2 not available, using V1")
        return None, None


class ScannerService:
    """
    Enhanced service for scanning option premiums.

    Integrates with UniverseService for:
    - Pre-filtering symbols by options availability
    - Sector/category filtering
    - Price and volume filtering
    - Symbol validation
    """

    def __init__(self):
        self._scanner_v1 = PremiumScanner()  # Fallback
        self._scanner_v2 = None
        self._universe = None
        self._universe_filter = None
        self._asset_type = None
        self._lock = threading.Lock()
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of V2 components"""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            # Try to load V2 scanner
            scanner, _ = _get_scanner_v2()
            if scanner:
                self._scanner_v2 = scanner

            # Try to load universe service
            universe, filter_cls, asset_type = _get_universe_service()
            if universe:
                self._universe = universe
                self._universe_filter = filter_cls
                self._asset_type = asset_type

            self._initialized = True

    def get_universe_stats(self) -> Dict[str, Any]:
        """Get statistics about available stocks and ETFs"""
        self._ensure_initialized()
        if self._universe:
            return self._universe.get_universe_stats()
        return {"error": "Universe service not available", "stocks": {}, "etfs": {}}

    def get_sectors(self) -> List[str]:
        """Get list of available sectors"""
        self._ensure_initialized()
        if self._universe:
            return self._universe.get_sectors()
        return []

    def get_categories(self) -> List[str]:
        """Get list of available ETF categories"""
        self._ensure_initialized()
        if self._universe:
            return self._universe.get_categories()
        return []

    def get_optionable_symbols(
        self,
        asset_type: str = "all",
        max_price: Optional[float] = None,
        sectors: Optional[List[str]] = None,
        limit: int = 500
    ) -> List[str]:
        """
        Get symbols that have options available from universe.

        Args:
            asset_type: "stock", "etf", or "all"
            max_price: Maximum stock price filter
            sectors: Filter to specific sectors
            limit: Maximum symbols to return

        Returns:
            List of optionable symbols
        """
        self._ensure_initialized()

        if not self._universe or not self._universe_filter:
            logger.warning("Universe not available, returning empty list")
            return []

        symbols = []

        if asset_type in ("stock", "all"):
            filter = self._universe_filter(
                max_price=max_price,
                sectors=sectors,
                has_options_only=True,
                active_only=True,
                limit=limit
            )
            stocks = self._universe.get_stocks(filter)
            symbols.extend(s.symbol for s in stocks)

        if asset_type in ("etf", "all"):
            filter = self._universe_filter(
                max_price=max_price,
                has_options_only=True,
                active_only=True,
                limit=limit // 2
            )
            etfs = self._universe.get_etfs(filter)
            symbols.extend(e.symbol for e in etfs)

        return symbols[:limit]

    def scan_premiums(
        self,
        symbols: List[str],
        max_price: float = 50.0,
        min_premium_pct: float = 1.0,
        dte: int = 30,
        sectors: Optional[List[str]] = None,
        use_cache: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Scan for premium opportunities.

        Args:
            symbols: List of stock symbols to scan
            max_price: Maximum stock price to consider
            min_premium_pct: Minimum premium as percentage of strike
            dte: Target days to expiration
            sectors: Filter to specific sectors
            use_cache: Use cached results
            progress_callback: Callback for progress updates

        Returns:
            List of premium opportunities sorted by monthly return
        """
        self._ensure_initialized()

        try:
            # Generate cache key
            if use_cache:
                cache_key = f"scan_{hash(frozenset(symbols))}_{max_price}_{min_premium_pct}_{dte}"
                cached = query_cache.get(cache_key)
                if cached:
                    logger.info("Returning cached scan results")
                    if sectors:
                        cached = [r for r in cached if r.get('sector') in sectors]
                    return cached

            # Use V2 scanner if available
            if self._scanner_v2:
                results = self._scanner_v2.scan_premiums(
                    symbols=symbols,
                    max_price=max_price,
                    min_premium_pct=min_premium_pct,
                    dte=dte,
                    sectors=sectors,
                    use_cache=use_cache,
                    progress_callback=progress_callback
                )
            else:
                # Fallback to V1 scanner
                results = self._scanner_v1.scan_premiums(
                    symbols=symbols,
                    max_price=max_price,
                    min_premium_pct=min_premium_pct,
                    dte=dte
                )

            # Cache for 5 minutes
            if use_cache and results:
                query_cache.set(cache_key, results, ttl_seconds=300)

            return results

        except Exception as e:
            logger.error(f"Error scanning premiums: {e}")
            raise

    def scan_from_universe(
        self,
        max_price: float = 100.0,
        min_premium_pct: float = 1.0,
        dte: int = 30,
        sectors: Optional[List[str]] = None,
        include_etfs: bool = True,
        min_volume: int = 100000,
        limit: int = 200,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Scan symbols from universe with automatic filtering.

        This is the recommended method for production scans as it:
        - Automatically filters for optionable symbols
        - Applies price and volume filters
        - Supports sector filtering
        - Includes ETFs optionally
        """
        self._ensure_initialized()

        # Get scannable symbols from universe
        if self._scanner_v2:
            symbols = self._scanner_v2.get_scannable_symbols(
                max_price=max_price,
                min_volume=min_volume,
                sectors=sectors,
                include_etfs=include_etfs,
                limit=limit
            )
        else:
            symbols = self.get_optionable_symbols(
                asset_type="all" if include_etfs else "stock",
                max_price=max_price,
                sectors=sectors,
                limit=limit
            )

        if not symbols:
            logger.warning("No scannable symbols found from universe")
            return []

        logger.info(f"Scanning {len(symbols)} symbols from universe")

        return self.scan_premiums(
            symbols=symbols,
            max_price=max_price,
            min_premium_pct=min_premium_pct,
            dte=dte,
            sectors=sectors,
            progress_callback=progress_callback
        )

    def scan_multiple_dte(
        self,
        symbols: List[str],
        max_price: float = 50.0,
        min_premium_pct: float = 1.0,
        dte_targets: Optional[List[int]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scan for premiums at multiple DTE targets.
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

    def get_dte_comparison(
        self,
        symbols: List[str],
        max_price: float = 50.0,
        min_premium_pct: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Get comparison stats across different DTEs."""
        dte_targets = [7, 14, 30, 45]
        results = self.scan_multiple_dte(
            symbols=symbols,
            max_price=max_price,
            min_premium_pct=min_premium_pct,
            dte_targets=dte_targets
        )

        comparison = []
        for dte in dte_targets:
            opps = results.get(str(dte), [])
            if opps:
                avg_monthly = sum(o.get('monthly_return', 0) for o in opps) / len(opps)
                avg_iv = sum(o.get('iv', 0) for o in opps) / len(opps)
                avg_premium_pct = sum(o.get('premium_pct', 0) for o in opps) / len(opps)
            else:
                avg_monthly = avg_iv = avg_premium_pct = 0

            comparison.append({
                'dte': dte,
                'opportunity_count': len(opps),
                'avg_monthly_return': round(avg_monthly, 2),
                'avg_iv': round(avg_iv, 1),
                'avg_premium_pct': round(avg_premium_pct, 2)
            })

        return comparison

    def get_quick_scan(self, dte: int = 30, limit: int = 20) -> List[Dict[str, Any]]:
        """Quick scan using universe-based symbol selection."""
        self._ensure_initialized()

        try:
            # Try to get symbols from universe first
            symbols = self.get_optionable_symbols(
                asset_type="stock",
                max_price=200,
                limit=50
            )

            if not symbols:
                # Fallback to hardcoded list
                symbols = [
                    'AAPL', 'AMD', 'AMZN', 'BAC', 'C', 'CCL', 'CSCO', 'F', 'GE',
                    'GOOG', 'INTC', 'META', 'MSFT', 'NVDA', 'PLTR', 'PYPL', 'SNAP',
                    'SOFI', 'T', 'TSLA', 'UAL', 'UBER', 'WFC', 'XOM'
                ]

            results = self.scan_premiums(
                symbols=symbols,
                max_price=200,
                min_premium_pct=0.5,
                dte=dte
            )
            return results[:limit]

        except Exception as e:
            logger.error(f"Error in quick scan: {e}")
            return []

    def scan_by_sector(
        self,
        sector: str,
        max_price: float = 100.0,
        min_premium_pct: float = 1.0,
        dte: int = 30,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Scan all optionable stocks in a specific sector."""
        self._ensure_initialized()

        if not self._universe or not self._universe_filter:
            logger.warning("Universe not available for sector scan")
            return []

        filter = self._universe_filter(
            max_price=max_price,
            sectors=[sector],
            has_options_only=True,
            active_only=True,
            limit=limit
        )

        stocks = self._universe.get_stocks(filter)
        symbols = [s.symbol for s in stocks]

        if not symbols:
            logger.warning(f"No optionable stocks found in sector: {sector}")
            return []

        return self.scan_premiums(
            symbols=symbols,
            max_price=max_price,
            min_premium_pct=min_premium_pct,
            dte=dte
        )

    def scan_etfs(
        self,
        categories: Optional[List[str]] = None,
        max_price: float = 100.0,
        min_premium_pct: float = 0.5,
        dte: int = 30,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Scan ETFs for premium opportunities."""
        self._ensure_initialized()

        if not self._universe or not self._universe_filter:
            logger.warning("Universe not available for ETF scan")
            return []

        filter = self._universe_filter(
            max_price=max_price,
            categories=categories,
            has_options_only=True,
            active_only=True,
            limit=limit
        )

        etfs = self._universe.get_etfs(filter)
        symbols = [e.symbol for e in etfs]

        if not symbols:
            logger.warning("No optionable ETFs found")
            return []

        return self.scan_premiums(
            symbols=symbols,
            max_price=max_price,
            min_premium_pct=min_premium_pct,
            dte=dte
        )

    def get_symbol_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a symbol from universe."""
        self._ensure_initialized()
        if self._universe:
            info = self._universe.get_symbol_info(symbol)
            return info.to_dict() if info else None
        return None

    def find_assignment_candidates(
        self,
        positions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find CSPs likely to be assigned."""
        try:
            return self._scanner_v1.find_assignment_candidates(positions)
        except Exception as e:
            logger.error(f"Error finding assignment candidates: {e}")
            return []

    def invalidate_cache(self) -> Dict[str, Any]:
        """Clear all caches."""
        self._ensure_initialized()
        result = {'timestamp': datetime.now().isoformat()}

        if self._scanner_v2:
            result['scanner_cache'] = self._scanner_v2.clear_cache()

        if self._universe:
            self._universe.invalidate_cache()
            result['universe_cache'] = 'cleared'

        return result


# Singleton instance
_scanner_service = None
_service_lock = threading.Lock()


def get_scanner_service() -> ScannerService:
    """Get scanner service singleton"""
    global _scanner_service
    if _scanner_service is None:
        with _service_lock:
            if _scanner_service is None:
                _scanner_service = ScannerService()
    return _scanner_service
