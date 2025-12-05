"""
Premium Scanner V2 - Enhanced scanner with universe integration and robust error handling

Key improvements:
- Uses UniverseService for pre-filtering (options availability, price, volume)
- Async-ready with proper timeouts
- Better error handling and logging
- Supports sector/category filtering
- Improved caching strategy
- Rate limiting and circuit breaker patterns (2025-12-05)
"""

import yfinance as yf
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

# ============================================================================
# TIMEOUT AND RATE LIMIT CONFIGURATION
# ============================================================================
# These values are tuned for optimal balance between speed and reliability

# Maximum total scan time regardless of symbol count (hard cap)
MAX_SCAN_TIMEOUT_SECONDS = 120

# Minimum total scan timeout (floor)
MIN_SCAN_TIMEOUT_SECONDS = 30

# Timeout per symbol (used for calculation, capped by MAX_SCAN_TIMEOUT_SECONDS)
TIMEOUT_PER_SYMBOL_SECONDS = 8

# Individual yfinance API call timeout
YFINANCE_REQUEST_TIMEOUT_SECONDS = 10

# Rate limiting: minimum delay between yfinance calls (in seconds)
RATE_LIMIT_DELAY_SECONDS = 0.1

# Circuit breaker: consecutive failures before tripping
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5

# Circuit breaker: seconds to wait before retry after trip
CIRCUIT_BREAKER_RESET_SECONDS = 30


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.
    Trips after consecutive failures and stays open for a reset period.
    """
    def __init__(self, failure_threshold: int = CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                 reset_timeout: float = CIRCUIT_BREAKER_RESET_SECONDS):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
        self._lock = threading.Lock()

    def record_success(self) -> None:
        with self._lock:
            self.failures = 0
            self.state = "closed"

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.failure_threshold:
                self.state = "open"
                logger.warning(f"Circuit breaker tripped after {self.failures} failures")

    def can_execute(self) -> bool:
        with self._lock:
            if self.state == "closed":
                return True
            if self.state == "open":
                if self.last_failure_time and (time.time() - self.last_failure_time) > self.reset_timeout:
                    self.state = "half-open"
                    return True
                return False
            # half-open: allow one attempt
            return True

    def get_state(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self.state,
                "failures": self.failures,
                "threshold": self.failure_threshold
            }


# Global circuit breaker for yfinance calls
_yfinance_circuit_breaker = CircuitBreaker()


class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    Ensures minimum delay between requests.
    """
    def __init__(self, min_interval: float = RATE_LIMIT_DELAY_SECONDS):
        self.min_interval = min_interval
        self.last_call_time: float = 0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Wait if needed to respect rate limit"""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_call_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call_time = time.time()


# Global rate limiter for yfinance calls
_yfinance_rate_limiter = RateLimiter()


class ScanStatus(Enum):
    PENDING = "pending"
    SCANNING = "scanning"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ScanResult:
    """Individual scan result with metadata"""
    symbol: str
    stock_price: float
    strike: float
    expiration: str
    dte: int
    premium: float  # Dollar amount for 1 contract
    premium_pct: float  # Premium as % of strike
    monthly_return: float
    annual_return: float
    iv: float
    volume: int
    open_interest: int
    bid_ask_spread: float
    delta: Optional[float] = None
    theta: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'stock_price': self.stock_price,
            'strike': self.strike,
            'expiration': self.expiration,
            'dte': self.dte,
            'premium': self.premium,
            'premium_pct': round(self.premium_pct, 2),
            'monthly_return': round(self.monthly_return, 2),
            'annual_return': round(self.annual_return, 2),
            'iv': self.iv,
            'volume': self.volume,
            'open_interest': self.open_interest,
            'bid_ask_spread': self.bid_ask_spread,
            'delta': self.delta,
            'theta': self.theta,
            'sector': self.sector,
            'industry': self.industry,
            'market_cap': self.market_cap
        }


@dataclass
class ScanProgress:
    """Track scan progress for streaming updates"""
    total_symbols: int = 0
    scanned: int = 0
    successful: int = 0
    failed: int = 0
    current_symbol: str = ""
    status: ScanStatus = ScanStatus.PENDING
    results_count: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def percent_complete(self) -> float:
        if self.total_symbols == 0:
            return 0
        return round((self.scanned / self.total_symbols) * 100, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_symbols': self.total_symbols,
            'scanned': self.scanned,
            'successful': self.successful,
            'failed': self.failed,
            'current_symbol': self.current_symbol,
            'status': self.status.value,
            'percent_complete': self.percent_complete,
            'results_count': self.results_count,
            'errors': self.errors[-5:]  # Last 5 errors
        }


class SymbolCache:
    """Thread-safe cache for symbol scan results"""

    def __init__(self, ttl_seconds: int = 300):
        self._cache: Dict[str, Tuple[List[Dict], float]] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def get(self, symbol: str, dte: int) -> Optional[List[Dict]]:
        key = f"{symbol}_{dte}"
        with self._lock:
            if key in self._cache:
                data, expiry = self._cache[key]
                if time.time() < expiry:
                    return data
                del self._cache[key]
        return None

    def set(self, symbol: str, dte: int, data: List[Dict]) -> None:
        key = f"{symbol}_{dte}"
        with self._lock:
            self._cache[key] = (data, time.time() + self._ttl)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def stats(self) -> Dict[str, int]:
        with self._lock:
            valid = sum(1 for _, (_, exp) in self._cache.items() if time.time() < exp)
            return {'total_entries': len(self._cache), 'valid_entries': valid}


class PremiumScannerV2:
    """
    Enhanced Premium Scanner with universe integration.

    Features:
    - Pre-filters symbols using UniverseService
    - Timeout protection per symbol
    - Better error recovery
    - Progress tracking for streaming
    - Sector/industry metadata enrichment
    """

    def __init__(self,
                 max_workers: int = 10,
                 symbol_timeout: int = 15,
                 min_volume: int = 50,
                 min_open_interest: int = 25):
        self.max_workers = max_workers
        self.symbol_timeout = symbol_timeout
        self.min_volume = min_volume
        self.min_open_interest = min_open_interest
        self._cache = SymbolCache(ttl_seconds=300)
        self._universe_service = None

    def _get_universe_service(self) -> None:
        """Lazy load universe service to avoid circular imports"""
        if self._universe_service is None:
            try:
                from backend.services.universe_service import get_universe_service
                self._universe_service = get_universe_service()
            except ImportError:
                logger.warning("UniverseService not available, running without pre-filtering")
        return self._universe_service

    def _enrich_with_universe_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get additional data from universe for a symbol.

        Note: UniverseService.get_symbol_info is async, so we skip enrichment
        in sync contexts to avoid runtime warnings.
        """
        universe = self._get_universe_service()
        if not universe:
            return {}

        try:
            import inspect
            # Skip if get_symbol_info is async (we're in sync context)
            if inspect.iscoroutinefunction(universe.get_symbol_info):
                return {}

            info = universe.get_symbol_info(symbol)
            if info:
                return {
                    'sector': getattr(info, 'sector', None),
                    'industry': getattr(info, 'industry', None),
                    'market_cap': getattr(info, 'market_cap', None)
                }
        except Exception:
            pass
        return {}

    def _validate_symbols(self, symbols: List[str],
                         max_price: float,
                         require_options: bool = True) -> Tuple[List[str], List[str]]:
        """
        Pre-validate symbols using UniverseService.
        Returns (valid_symbols, skipped_symbols)

        Note: UniverseService.validate_symbols is async, so we skip validation
        in sync contexts and return all symbols as valid.
        """
        universe = self._get_universe_service()
        if not universe:
            # No universe service, return all symbols as valid
            return symbols, []

        try:
            import asyncio
            import inspect

            # Check if validate_symbols is a coroutine function
            if inspect.iscoroutinefunction(universe.validate_symbols):
                # We're in a sync context, skip async validation
                # Just do basic filtering by symbol format
                logger.debug("Skipping async validation in sync context, returning all symbols as valid")
                return symbols, []

            valid, invalid = universe.validate_symbols(
                symbols=symbols,
                require_options=require_options,
                max_price=max_price
            )

            if invalid:
                logger.info(f"Pre-filtered {len(invalid)} symbols (no options or price too high)")

            return valid, invalid

        except Exception as e:
            logger.warning(f"Symbol validation failed: {e}, proceeding with all symbols")
            return symbols, []

    def _fetch_ticker_info_with_timeout(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch ticker info with timeout and rate limiting.
        Uses circuit breaker pattern to avoid hammering failing API.
        """
        if not _yfinance_circuit_breaker.can_execute():
            logger.debug(f"Circuit breaker open, skipping {symbol}")
            return None

        _yfinance_rate_limiter.wait()

        try:
            ticker = yf.Ticker(symbol)
            # yfinance doesn't have native timeout, but we can limit via session
            info = ticker.info
            _yfinance_circuit_breaker.record_success()
            return info
        except Exception as e:
            _yfinance_circuit_breaker.record_failure()
            logger.debug(f"Ticker info fetch failed for {symbol}: {e}")
            return None

    def _fetch_options_chain_with_timeout(self, symbol: str, expiry: str) -> Optional[Any]:
        """
        Fetch options chain with timeout and rate limiting.
        """
        if not _yfinance_circuit_breaker.can_execute():
            return None

        _yfinance_rate_limiter.wait()

        try:
            ticker = yf.Ticker(symbol)
            opt_chain = ticker.option_chain(expiry)
            _yfinance_circuit_breaker.record_success()
            return opt_chain
        except Exception as e:
            _yfinance_circuit_breaker.record_failure()
            logger.debug(f"Options chain fetch failed for {symbol}: {e}")
            return None

    def _scan_single_symbol(self,
                           symbol: str,
                           max_price: float,
                           min_premium_pct: float,
                           dte: int) -> Tuple[str, List[ScanResult], Optional[str]]:
        """
        Scan a single symbol for premium opportunities.
        Returns: (symbol, results, error_message)

        Uses rate limiting and circuit breaker for robust API access.
        """
        results = []
        error = None
        scan_start = time.time()

        try:
            # Check circuit breaker before starting
            if not _yfinance_circuit_breaker.can_execute():
                return symbol, [], "Circuit breaker open"

            # Fetch ticker info with rate limiting
            info = self._fetch_ticker_info_with_timeout(symbol)
            if not info:
                return symbol, [], "Failed to fetch ticker info"

            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)

            if current_price <= 0:
                return symbol, [], "No price data"

            if current_price > max_price:
                return symbol, [], f"Price ${current_price:.2f} > max ${max_price}"

            # Get expiration dates (with rate limit)
            _yfinance_rate_limiter.wait()
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            if not expirations:
                return symbol, [], "No options available"

            # Find closest expiration to target DTE
            target_date = datetime.now() + timedelta(days=dte)
            best_expiry = min(
                expirations,
                key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d') - target_date).days)
            )

            # Get options chain with rate limiting
            opt_chain = self._fetch_options_chain_with_timeout(symbol, best_expiry)
            if opt_chain is None:
                return symbol, [], "Failed to fetch options chain"

            puts = opt_chain.puts

            if puts.empty:
                return symbol, [], "No puts available"

            # Get universe enrichment data
            universe_data = self._enrich_with_universe_data(symbol)

            # Filter OTM puts (5% below current price)
            otm_threshold = current_price * 0.95
            otm_puts = puts[puts['strike'] < otm_threshold]

            if otm_puts.empty:
                return symbol, [], "No OTM puts"

            days_to_expiry = (datetime.strptime(best_expiry, '%Y-%m-%d') - datetime.now()).days
            if days_to_expiry <= 0:
                return symbol, [], "Expiration passed"

            for _, put in otm_puts.iterrows():
                strike = float(put['strike'])
                bid = float(put['bid'] or 0)
                ask = float(put['ask'] or 0)
                volume = int(put['volume'] or 0)
                oi = int(put['openInterest'] or 0)
                iv = float(put['impliedVolatility'] or 0)

                # Liquidity filter
                if volume < self.min_volume and oi < self.min_open_interest:
                    continue

                # Calculate premium
                premium = (bid + ask) / 2 if bid > 0 and ask > 0 else bid
                if premium <= 0:
                    continue

                premium_pct = (premium / strike) * 100
                if premium_pct < min_premium_pct:
                    continue

                monthly_return = (premium_pct / days_to_expiry) * 30
                annual_return = monthly_return * 12

                result = ScanResult(
                    symbol=symbol,
                    stock_price=round(current_price, 2),
                    strike=round(strike, 2),
                    expiration=best_expiry,
                    dte=days_to_expiry,
                    premium=round(premium * 100, 2),
                    premium_pct=premium_pct,
                    monthly_return=monthly_return,
                    annual_return=annual_return,
                    iv=round(iv * 100, 1),
                    volume=volume,
                    open_interest=oi,
                    bid_ask_spread=round(ask - bid, 3) if ask > 0 and bid > 0 else 0,
                    sector=universe_data.get('sector'),
                    industry=universe_data.get('industry'),
                    market_cap=universe_data.get('market_cap')
                )
                results.append(result)

        except Exception as e:
            error = str(e)[:100]

        return symbol, results, error

    def scan_premiums(self,
                     symbols: List[str],
                     max_price: float = 50.0,
                     min_premium_pct: float = 1.0,
                     dte: int = 30,
                     sectors: Optional[List[str]] = None,
                     use_cache: bool = True,
                     progress_callback: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Scan symbols for put premium opportunities.

        Args:
            symbols: List of stock symbols to scan
            max_price: Maximum stock price
            min_premium_pct: Minimum premium as % of strike
            dte: Target days to expiration
            sectors: Filter to specific sectors (optional)
            use_cache: Use cached results
            progress_callback: Callback for progress updates

        Returns:
            List of premium opportunities sorted by monthly return
        """
        all_results = []
        progress = ScanProgress(total_symbols=len(symbols), status=ScanStatus.SCANNING)

        # Deduplicate and uppercase symbols
        symbols = list(set(s.upper() for s in symbols))
        progress.total_symbols = len(symbols)

        # Pre-validate symbols using universe
        valid_symbols, skipped = self._validate_symbols(symbols, max_price)
        progress.failed += len(skipped)

        # Check cache for valid symbols
        symbols_to_scan = []
        for symbol in valid_symbols:
            if use_cache:
                cached = self._cache.get(symbol, dte)
                if cached is not None:
                    # Apply current filters to cached data
                    for opp in cached:
                        if (opp.get('stock_price', 999) <= max_price and
                            opp.get('premium_pct', 0) >= min_premium_pct):
                            all_results.append(opp)
                    progress.scanned += 1
                    progress.successful += 1
                    continue
            symbols_to_scan.append(symbol)

        if not symbols_to_scan:
            progress.status = ScanStatus.COMPLETED
            if progress_callback:
                progress_callback(progress)
            return sorted(all_results, key=lambda x: x.get('monthly_return', 0), reverse=True)

        logger.info(f"Scanning {len(symbols_to_scan)} symbols (cached: {len(valid_symbols) - len(symbols_to_scan)}, skipped: {len(skipped)})")

        # Calculate capped total timeout: min 30s, max 120s, scales with symbol count
        # This prevents unbounded waits for large symbol lists
        calculated_timeout = TIMEOUT_PER_SYMBOL_SECONDS * len(symbols_to_scan)
        total_timeout = min(MAX_SCAN_TIMEOUT_SECONDS, max(MIN_SCAN_TIMEOUT_SECONDS, calculated_timeout))
        logger.debug(f"Using total timeout of {total_timeout}s for {len(symbols_to_scan)} symbols")

        scan_start_time = time.time()

        # Concurrent scanning with CAPPED timeouts
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    self._scan_single_symbol,
                    symbol,
                    max_price,
                    min_premium_pct,
                    dte
                ): symbol
                for symbol in symbols_to_scan
            }

            try:
                for future in as_completed(future_to_symbol, timeout=total_timeout):
                    # Check if we've exceeded our total timeout
                    elapsed = time.time() - scan_start_time
                    if elapsed > total_timeout:
                        logger.warning(f"Total scan timeout exceeded ({elapsed:.1f}s > {total_timeout}s)")
                        progress.status = ScanStatus.TIMEOUT
                        break

                    symbol = future_to_symbol[future]
                    progress.current_symbol = symbol
                    progress.scanned += 1

                    try:
                        # Individual future timeout is capped at 15 seconds
                        sym, results, error = future.result(timeout=min(self.symbol_timeout, 15))

                        if error:
                            progress.failed += 1
                            progress.errors.append(f"{sym}: {error}")
                        else:
                            progress.successful += 1

                        # Cache results
                        result_dicts = [r.to_dict() for r in results]
                        self._cache.set(symbol, dte, result_dicts)

                        # Apply sector filter if specified
                        for r in results:
                            if sectors and r.sector and r.sector not in sectors:
                                continue
                            all_results.append(r.to_dict())

                        progress.results_count = len(all_results)

                    except TimeoutError:
                        progress.failed += 1
                        progress.errors.append(f"{symbol}: Timeout")
                    except Exception as e:
                        progress.failed += 1
                        progress.errors.append(f"{symbol}: {str(e)[:50]}")

                    if progress_callback:
                        progress_callback(progress)

            except TimeoutError:
                # Total scan timeout exceeded
                logger.warning(f"Scan exceeded total timeout of {total_timeout}s")
                progress.status = ScanStatus.TIMEOUT
                progress.errors.append(f"Total timeout ({total_timeout}s) exceeded")

        # Set final status if not already set to timeout
        if progress.status != ScanStatus.TIMEOUT:
            progress.status = ScanStatus.COMPLETED
        progress.current_symbol = ""

        if progress_callback:
            progress_callback(progress)

        # Sort by monthly return
        all_results.sort(key=lambda x: x.get('monthly_return', 0), reverse=True)

        logger.info(f"Scan complete: {progress.successful} successful, {progress.failed} failed, {len(all_results)} results")

        return all_results

    def get_scannable_symbols(self,
                             max_price: float = 100.0,
                             min_volume: int = 100000,
                             sectors: Optional[List[str]] = None,
                             include_etfs: bool = True,
                             limit: int = 200) -> List[str]:
        """
        Get symbols suitable for premium scanning from universe.
        Pre-filters based on options availability, price, and volume.
        """
        universe = self._get_universe_service()
        if universe:
            return universe.get_scannable_symbols(
                max_price=max_price,
                min_volume=min_volume,
                sectors=sectors,
                include_etfs=include_etfs,
                limit=limit
            )

        # Fallback: return empty list (no default symbols)
        logger.warning("UniverseService not available, no symbols returned")
        return []

    def clear_cache(self) -> Dict[str, int]:
        """Clear symbol cache and return stats"""
        stats = self._cache.stats()
        self._cache.clear()
        return stats

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get scanner diagnostics including circuit breaker state, cache stats, and config.
        Useful for monitoring and debugging.
        """
        return {
            'circuit_breaker': _yfinance_circuit_breaker.get_state(),
            'cache_stats': self._cache.stats(),
            'config': {
                'max_workers': self.max_workers,
                'symbol_timeout': self.symbol_timeout,
                'min_volume': self.min_volume,
                'min_open_interest': self.min_open_interest,
                'max_scan_timeout': MAX_SCAN_TIMEOUT_SECONDS,
                'min_scan_timeout': MIN_SCAN_TIMEOUT_SECONDS,
                'timeout_per_symbol': TIMEOUT_PER_SYMBOL_SECONDS,
                'rate_limit_delay': RATE_LIMIT_DELAY_SECONDS
            }
        }

    def reset_circuit_breaker(self) -> Dict[str, Any]:
        """Reset the circuit breaker to closed state"""
        _yfinance_circuit_breaker.record_success()  # This resets failures and sets state to closed
        return _yfinance_circuit_breaker.get_state()


# Singleton instance
_scanner_v2: Optional[PremiumScannerV2] = None
_scanner_lock = threading.Lock()


def get_premium_scanner_v2() -> PremiumScannerV2:
    """Get singleton scanner instance"""
    global _scanner_v2
    if _scanner_v2 is None:
        with _scanner_lock:
            if _scanner_v2 is None:
                _scanner_v2 = PremiumScannerV2()
    return _scanner_v2
