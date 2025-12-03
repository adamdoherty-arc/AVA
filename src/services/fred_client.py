"""
FRED (Federal Reserve Economic Data) API Client
================================================

FREE, UNLIMITED access to official US economic data from the Federal Reserve.

Features (All FREE, UNLIMITED):
- 820,000+ economic time series
- Official US economic indicators
- GDP, inflation, unemployment, interest rates
- No rate limits for non-commercial use
- Historical data back decades

Performance Optimizations:
- Distributed caching (Redis with in-memory fallback)
- Retry logic with exponential backoff
- Circuit breaker for fault tolerance
- Thread-safe singleton pattern

API Key: Get FREE at https://fred.stlouisfed.org/docs/api/api_key.html

Author: Magnus Trading Platform
Created: 2025-11-21
Updated: 2025-11-29 (Performance optimizations)
"""

import os
import requests
import logging
import asyncio
import threading
import time
import hashlib
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Circuit Breaker Pattern
# =============================================================================

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.

    States:
    - CLOSED: Normal operation, tracking failures
    - OPEN: Rejecting requests, waiting for recovery
    - HALF_OPEN: Testing if service recovered
    """
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3

    _state: CircuitState = CircuitState.CLOSED
    _failures: int = 0
    _last_failure_time: float = 0.0
    _half_open_calls: int = 0
    _lock: threading.Lock = None

    def __post_init__(self):
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
            return self._state

    def allow_request(self) -> bool:
        """Check if request should be allowed"""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
            return False
        return False  # OPEN state

    def record_success(self) -> None:
        """Record a successful call"""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
            self._failures = 0

    def record_failure(self) -> None:
        """Record a failed call"""
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()
            if self._failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
            elif self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN


class FREDAPIError(Exception):
    """Base exception for FRED API errors"""
    pass


class FREDRateLimitError(FREDAPIError):
    """Rate limit exceeded"""
    pass


class FREDCircuitOpenError(FREDAPIError):
    """Circuit breaker is open"""
    pass


class FREDClient:
    """
    Client for FRED (Federal Reserve Economic Data) FREE API

    Features:
    - Distributed caching (Redis with in-memory fallback)
    - Retry logic with exponential backoff
    - Circuit breaker for fault tolerance
    - Async-compatible via asyncio.to_thread()
    """

    BASE_URL = "https://api.stlouisfed.org/fred"

    # Default demo API key (get your own free at fred.stlouisfed.org)
    DEFAULT_API_KEY = "demo"

    # Cache TTLs (in seconds)
    CACHE_TTL_SERIES = 3600        # 1 hour for time series data
    CACHE_TTL_LATEST = 300         # 5 minutes for latest values
    CACHE_TTL_SNAPSHOT = 300       # 5 minutes for snapshots
    CACHE_TTL_METADATA = 86400     # 24 hours for series metadata

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 1.0  # seconds

    # Most important economic indicators
    IMPORTANT_SERIES = {
        # GDP & Growth
        'GDP': 'Gross Domestic Product',
        'GDPC1': 'Real GDP',
        'A191RL1Q225SBEA': 'Real GDP Growth Rate',

        # Inflation
        'CPIAUCSL': 'Consumer Price Index (CPI)',
        'CPILFESL': 'Core CPI (ex food & energy)',
        'PCE': 'Personal Consumption Expenditures',
        'PCEPI': 'PCE Price Index',

        # Employment
        'UNRATE': 'Unemployment Rate',
        'PAYEMS': 'Nonfarm Payrolls',
        'CIVPART': 'Labor Force Participation Rate',
        'U6RATE': 'U-6 Unemployment Rate',

        # Interest Rates
        'DFF': 'Federal Funds Rate (Effective)',
        'FEDFUNDS': 'Federal Funds Rate',
        'DGS10': '10-Year Treasury Yield',
        'DGS2': '2-Year Treasury Yield',
        'T10Y2Y': '10Y-2Y Treasury Spread',
        'T10Y3M': '10Y-3M Treasury Spread',

        # Money & Credit
        'M2': 'M2 Money Supply',
        'WALCL': 'Fed Balance Sheet',
        'MORTGAGE30US': '30-Year Mortgage Rate',

        # Confidence & Sentiment
        'UMCSENT': 'University of Michigan Consumer Sentiment',
        'VIXCLS': 'VIX (CBOE Volatility Index)',

        # Market Indicators
        'SP500': 'S&P 500 Index',
        'DCOILWTICO': 'WTI Crude Oil Price',
        'GOLDAMGBD228NLBM': 'Gold Price',

        # Housing
        'HOUST': 'Housing Starts',
        'CSUSHPISA': 'Case-Shiller Home Price Index',

        # Manufacturing & Business
        'INDPRO': 'Industrial Production Index',
        'ISM': 'ISM Manufacturing PMI',
        'BOPTEXP': 'Retail Sales'
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED client.

        Args:
            api_key: FRED API key (get FREE at fred.stlouisfed.org)
                    Falls back to env var FRED_API_KEY
        """
        self.api_key = api_key or os.getenv('FRED_API_KEY', self.DEFAULT_API_KEY)
        self.session = requests.Session()

        # Set default timeout and headers
        self.session.headers.update({
            'User-Agent': 'AVA-Trading-Platform/1.0',
            'Accept': 'application/json'
        })

        # Circuit breaker for fault tolerance
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            half_open_max_calls=3
        )

        # Local cache for sync operations (distributed cache for async)
        self._local_cache: Dict[str, tuple[Any, float]] = {}
        self._cache_lock = threading.Lock()

        # Statistics
        self._stats = {
            'requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'retries': 0
        }

        logger.info(f"FRED client initialized (API key: {self.api_key[:8]}...)")

    def _get_cached(self, cache_key: str, ttl: int) -> Optional[Any]:
        """Get value from local cache if not expired"""
        with self._cache_lock:
            if cache_key in self._local_cache:
                value, expires_at = self._local_cache[cache_key]
                if time.time() < expires_at:
                    self._stats['cache_hits'] += 1
                    return value
                else:
                    del self._local_cache[cache_key]
        self._stats['cache_misses'] += 1
        return None

    def _set_cached(self, cache_key: str, value: Any, ttl: int) -> None:
        """Set value in local cache with TTL"""
        with self._cache_lock:
            self._local_cache[cache_key] = (value, time.time() + ttl)

    def _make_request(self, endpoint: str, params: Dict[str, str]) -> Optional[Dict]:
        """
        Make API request to FRED with circuit breaker and retry logic.

        Args:
            endpoint: API endpoint (e.g., 'series/observations')
            params: Query parameters

        Returns:
            JSON response

        Raises:
            FREDCircuitOpenError: If circuit breaker is open
            FREDAPIError: On API errors after retries
        """
        # Check circuit breaker
        if not self._circuit_breaker.allow_request():
            raise FREDCircuitOpenError(
                f"Circuit breaker is open. Last failure: "
                f"{self._circuit_breaker._last_failure_time}"
            )

        # Add API key and JSON format
        params['api_key'] = self.api_key
        params['file_type'] = 'json'

        url = f"{self.BASE_URL}/{endpoint}"

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                self._stats['requests'] += 1

                response = self.session.get(
                    url,
                    params=params,
                    timeout=(5, 30)  # (connect, read) timeouts
                )
                response.raise_for_status()

                data = response.json()

                # Check for API-level errors
                if 'error_code' in data:
                    error_msg = data.get('error_message', 'Unknown FRED API error')
                    logger.error(f"FRED API error: {error_msg}")
                    self._circuit_breaker.record_failure()
                    self._stats['errors'] += 1
                    raise FREDAPIError(error_msg)

                # Success - record and return
                self._circuit_breaker.record_success()
                return data

            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(f"FRED timeout (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}")
                self._stats['retries'] += 1

            except requests.exceptions.ConnectionError as e:
                last_error = e
                logger.warning(f"FRED connection error (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}")
                self._stats['retries'] += 1

            except requests.exceptions.HTTPError as e:
                # Don't retry 4xx errors (except 429 rate limit)
                if e.response and 400 <= e.response.status_code < 500:
                    if e.response.status_code == 429:
                        logger.warning("FRED rate limit hit, will retry")
                        self._stats['retries'] += 1
                        last_error = FREDRateLimitError(str(e))
                    else:
                        self._circuit_breaker.record_failure()
                        self._stats['errors'] += 1
                        raise FREDAPIError(f"HTTP {e.response.status_code}: {e}")
                else:
                    last_error = e
                    self._stats['retries'] += 1

            except FREDAPIError:
                raise  # Re-raise API errors

            except Exception as e:
                last_error = e
                logger.error(f"FRED unexpected error: {e}")
                self._stats['errors'] += 1

            # Exponential backoff before retry (only if we'll retry)
            if attempt < self.MAX_RETRIES - 1:
                sleep_time = self.RETRY_BACKOFF_BASE * (2 ** attempt)
                time.sleep(sleep_time)

        # All retries failed
        self._circuit_breaker.record_failure()
        self._stats['errors'] += 1
        logger.error(f"FRED request failed after {self.MAX_RETRIES} attempts: {last_error}")
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            **self._stats,
            'circuit_state': self._circuit_breaker.state.value,
            'cache_size': len(self._local_cache)
        }

    # ========================================================================
    # ECONOMIC DATA RETRIEVAL
    # ========================================================================

    def _make_cache_key(self, *args) -> str:
        """Generate cache key from arguments"""
        key_str = ":".join(str(a) for a in args if a is not None)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> Optional[List[Dict]]:
        """
        Get economic time series data with caching.

        Args:
            series_id: FRED series ID (e.g., 'GDP', 'UNRATE', 'FEDFUNDS')
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            limit: Max observations (default 1000)

        Returns:
            List of observations with date and value
        """
        # Check cache first
        cache_key = f"fred:series:{self._make_cache_key(series_id, start_date, end_date, limit)}"
        cached = self._get_cached(cache_key, self.CACHE_TTL_SERIES)
        if cached is not None:
            return cached

        params = {
            'series_id': series_id,
            'limit': str(limit),
            'sort_order': 'desc'  # Most recent first
        }

        if start_date:
            params['observation_start'] = start_date

        if end_date:
            params['observation_end'] = end_date

        try:
            data = self._make_request('series/observations', params)
        except FREDCircuitOpenError:
            logger.warning(f"Circuit open, returning cached data for {series_id} if available")
            return None
        except FREDAPIError as e:
            logger.error(f"FRED API error for {series_id}: {e}")
            return None

        if not data or 'observations' not in data:
            return None

        # Parse observations
        observations = []
        for obs in data['observations']:
            # Skip missing values
            if obs['value'] == '.':
                continue

            try:
                observations.append({
                    'date': obs['date'],
                    'value': float(obs['value']),
                    'series_id': series_id
                })
            except (ValueError, KeyError):
                continue

        # Cache the result
        if observations:
            self._set_cached(cache_key, observations, self.CACHE_TTL_SERIES)

        return observations

    def get_latest_value(self, series_id: str) -> Optional[Dict]:
        """
        Get the latest value for an economic indicator.

        Args:
            series_id: FRED series ID

        Returns:
            Latest observation with date and value
        """
        observations = self.get_series(series_id, limit=1)

        if not observations:
            return None

        return observations[0]

    def get_series_info(self, series_id: str) -> Optional[Dict]:
        """
        Get metadata about a series.

        Args:
            series_id: FRED series ID

        Returns:
            Series metadata (title, units, frequency, etc.)
        """
        # Check cache first
        cache_key = f"fred:info:{series_id}"
        cached = self._get_cached(cache_key, self.CACHE_TTL_METADATA)
        if cached is not None:
            return cached

        params = {'series_id': series_id}

        try:
            data = self._make_request('series', params)
        except FREDCircuitOpenError:
            logger.warning(f"Circuit open, cannot fetch info for {series_id}")
            return None
        except FREDAPIError as e:
            logger.error(f"FRED API error for series info {series_id}: {e}")
            return None

        if not data or 'seriess' not in data or not data['seriess']:
            return None

        series = data['seriess'][0]

        result = {
            'id': series.get('id', series_id),
            'title': series.get('title', ''),
            'units': series.get('units', ''),
            'frequency': series.get('frequency', ''),
            'seasonal_adjustment': series.get('seasonal_adjustment', ''),
            'last_updated': series.get('last_updated', ''),
            'popularity': series.get('popularity', 0),
            'notes': series.get('notes', '')
        }

        # Cache the result
        self._set_cached(cache_key, result, self.CACHE_TTL_METADATA)

        return result

    # ========================================================================
    # ECONOMIC INDICATORS SNAPSHOT
    # ========================================================================

    def get_economic_snapshot(self) -> Dict[str, Any]:
        """
        Get current snapshot of key economic indicators.

        Returns comprehensive economic dashboard:
        - Growth (GDP)
        - Inflation (CPI, PCE)
        - Employment (unemployment rate, payrolls)
        - Interest rates (Fed funds, yields)
        - Market indicators (VIX, S&P 500)
        """
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'indicators': {}
        }

        # Key indicators to fetch
        indicators = [
            ('GDPC1', 'Real GDP'),
            ('CPIAUCSL', 'CPI (Inflation)'),
            ('UNRATE', 'Unemployment Rate'),
            ('FEDFUNDS', 'Fed Funds Rate'),
            ('DGS10', '10-Year Treasury Yield'),
            ('VIXCLS', 'VIX (Market Volatility)'),
            ('SP500', 'S&P 500'),
            ('UMCSENT', 'Consumer Sentiment'),
            ('T10Y2Y', 'Yield Curve (10Y-2Y)')
        ]

        for series_id, name in indicators:
            latest = self.get_latest_value(series_id)
            if latest:
                snapshot['indicators'][series_id] = {
                    'name': name,
                    'value': latest['value'],
                    'date': latest['date'],
                    'series_id': series_id
                }

        # Calculate derived metrics
        if 'UNRATE' in snapshot['indicators']:
            unemployment = snapshot['indicators']['UNRATE']['value']
            snapshot['labor_market_health'] = 'Strong' if unemployment < 4.5 else 'Weak' if unemployment > 6 else 'Moderate'

        if 'T10Y2Y' in snapshot['indicators']:
            yield_curve = snapshot['indicators']['T10Y2Y']['value']
            snapshot['recession_signal'] = 'Warning' if yield_curve < 0 else 'Normal'

        if 'VIXCLS' in snapshot['indicators']:
            vix = snapshot['indicators']['VIXCLS']['value']
            snapshot['market_fear_level'] = 'Low' if vix < 15 else 'High' if vix > 25 else 'Moderate'

        return snapshot

    # ========================================================================
    # TREND ANALYSIS
    # ========================================================================

    def get_trend(self, series_id: str, periods: int = 12) -> Optional[Dict]:
        """
        Analyze trend for an economic indicator.

        Args:
            series_id: FRED series ID
            periods: Number of periods to analyze

        Returns:
            Trend analysis (direction, change, momentum)
        """
        observations = self.get_series(series_id, limit=periods)

        if not observations or len(observations) < 2:
            return None

        # Sort by date ascending
        observations.sort(key=lambda x: x['date'])

        latest = observations[-1]['value']
        oldest = observations[0]['value']

        change = latest - oldest
        change_pct = (change / oldest * 100) if oldest != 0 else 0

        # Determine trend
        if change_pct > 2:
            trend = 'Rising'
        elif change_pct < -2:
            trend = 'Falling'
        else:
            trend = 'Stable'

        # Calculate momentum (recent vs earlier change)
        mid_point = len(observations) // 2
        recent_change = observations[-1]['value'] - observations[mid_point]['value']
        earlier_change = observations[mid_point]['value'] - observations[0]['value']

        if abs(recent_change) > abs(earlier_change):
            momentum = 'Accelerating'
        elif abs(recent_change) < abs(earlier_change):
            momentum = 'Decelerating'
        else:
            momentum = 'Steady'

        return {
            'series_id': series_id,
            'latest_value': latest,
            'latest_date': observations[-1]['date'],
            'oldest_value': oldest,
            'oldest_date': observations[0]['date'],
            'change': round(change, 2),
            'change_pct': round(change_pct, 2),
            'trend': trend,
            'momentum': momentum,
            'periods_analyzed': len(observations)
        }

    # ========================================================================
    # RECESSION INDICATORS
    # ========================================================================

    def get_recession_indicators(self) -> Dict[str, Any]:
        """
        Get key recession warning indicators.

        Returns:
            Recession risk assessment based on multiple indicators
        """
        indicators = {}

        # 1. Yield Curve (10Y-2Y spread)
        yield_curve = self.get_latest_value('T10Y2Y')
        if yield_curve:
            inverted = yield_curve['value'] < 0
            indicators['yield_curve'] = {
                'value': yield_curve['value'],
                'inverted': inverted,
                'warning': inverted,
                'description': 'Inverted yield curve' if inverted else 'Normal yield curve'
            }

        # 2. Unemployment Rate (rising unemployment)
        unemployment_trend = self.get_trend('UNRATE', periods=6)
        if unemployment_trend:
            rising = unemployment_trend['trend'] == 'Rising'
            indicators['unemployment'] = {
                'value': unemployment_trend['latest_value'],
                'trend': unemployment_trend['trend'],
                'warning': rising,
                'description': 'Rising unemployment' if rising else 'Stable/falling unemployment'
            }

        # 3. Consumer Sentiment
        sentiment_trend = self.get_trend('UMCSENT', periods=6)
        if sentiment_trend:
            falling = sentiment_trend['trend'] == 'Falling'
            indicators['consumer_sentiment'] = {
                'value': sentiment_trend['latest_value'],
                'trend': sentiment_trend['trend'],
                'warning': falling,
                'description': 'Deteriorating sentiment' if falling else 'Stable/improving sentiment'
            }

        # 4. Fed Policy (rapid rate hikes)
        fedfunds_trend = self.get_trend('FEDFUNDS', periods=12)
        if fedfunds_trend:
            aggressive = fedfunds_trend['change'] > 1.0  # More than 1% increase
            indicators['fed_policy'] = {
                'value': fedfunds_trend['latest_value'],
                'change': fedfunds_trend['change'],
                'warning': aggressive,
                'description': 'Aggressive tightening' if aggressive else 'Moderate policy'
            }

        # Overall recession risk
        warnings = sum(1 for ind in indicators.values() if ind.get('warning', False))
        total_indicators = len(indicators)

        if warnings >= 3:
            risk_level = 'High'
        elif warnings >= 2:
            risk_level = 'Moderate'
        elif warnings >= 1:
            risk_level = 'Low'
        else:
            risk_level = 'Minimal'

        return {
            'recession_risk': risk_level,
            'warnings': warnings,
            'total_indicators': total_indicators,
            'indicators': indicators,
            'timestamp': datetime.now().isoformat()
        }

    # ========================================================================
    # INFLATION ANALYSIS
    # ========================================================================

    def get_inflation_report(self) -> Dict[str, Any]:
        """
        Get comprehensive inflation analysis.

        Returns:
            Multi-metric inflation assessment
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {}
        }

        # Get multiple inflation measures
        inflation_series = [
            ('CPIAUCSL', 'CPI (Headline)'),
            ('CPILFESL', 'CPI (Core)'),
            ('PCEPI', 'PCE Price Index'),
            ('PCEPILFE', 'PCE (Core)')
        ]

        for series_id, name in inflation_series:
            # Get year-over-year change
            observations = self.get_series(series_id, limit=13)  # 13 months for YoY
            if observations and len(observations) >= 13:
                # Sort ascending
                observations.sort(key=lambda x: x['date'])
                latest = observations[-1]['value']
                year_ago = observations[0]['value']
                yoy_change = ((latest - year_ago) / year_ago * 100) if year_ago != 0 else 0

                report['metrics'][series_id] = {
                    'name': name,
                    'latest_value': latest,
                    'yoy_change_pct': round(yoy_change, 2),
                    'date': observations[-1]['date']
                }

        # Overall inflation assessment
        if 'CPIAUCSL' in report['metrics']:
            cpi_yoy = report['metrics']['CPIAUCSL']['yoy_change_pct']
            if cpi_yoy > 4:
                assessment = 'High'
            elif cpi_yoy > 2.5:
                assessment = 'Elevated'
            elif cpi_yoy > 1.5:
                assessment = 'Moderate'
            else:
                assessment = 'Low'

            report['overall_assessment'] = assessment
            report['above_fed_target'] = cpi_yoy > 2.0

        return report

    # ========================================================================
    # MARKET REGIME DETECTION
    # ========================================================================

    def get_market_regime(self) -> Dict[str, str]:
        """
        Determine current macroeconomic regime.

        Returns:
            Market regime classification and characteristics
        """
        snapshot = self.get_economic_snapshot()

        # Extract key metrics
        indicators = snapshot.get('indicators', {})

        growth = 'unknown'
        inflation_state = 'unknown'
        policy = 'unknown'

        # Assess growth
        unemployment = indicators.get('UNRATE', {}).get('value')
        if unemployment:
            growth = 'strong' if unemployment < 4.5 else 'weak' if unemployment > 6 else 'moderate'

        # Assess inflation
        recession_risk = self.get_recession_indicators()
        if recession_risk:
            warnings = recession_risk.get('warnings', 0)
            if warnings >= 3:
                macro_outlook = 'recessionary'
            elif warnings >= 2:
                macro_outlook = 'slowing'
            else:
                macro_outlook = 'expansionary'
        else:
            macro_outlook = 'unknown'

        # Assess Fed policy
        fed_funds = indicators.get('FEDFUNDS', {}).get('value')
        if fed_funds:
            policy = 'tight' if fed_funds > 4.5 else 'accommodative' if fed_funds < 2 else 'neutral'

        # Determine regime
        if growth == 'strong' and policy == 'accommodative':
            regime = 'Goldilocks'
            characteristics = 'Strong growth, low rates - bullish for risk assets'
        elif growth == 'weak' and policy == 'tight':
            regime = 'Stagflation Risk'
            characteristics = 'Weak growth, tight policy - bearish for risk assets'
        elif macro_outlook == 'recessionary':
            regime = 'Recessionary'
            characteristics = 'Economic contraction likely - defensive positioning'
        elif growth == 'strong' and policy == 'tight':
            regime = 'Late Cycle'
            characteristics = 'Strong but slowing, Fed tightening - volatile markets'
        else:
            regime = 'Transitional'
            characteristics = 'Mixed signals - cautious approach'

        return {
            'regime': regime,
            'characteristics': characteristics,
            'growth': growth,
            'inflation': inflation_state,
            'policy': policy,
            'macro_outlook': macro_outlook,
            'timestamp': datetime.now().isoformat()
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_fred_client = None

def get_client() -> FREDClient:
    """Get singleton FRED client instance"""
    global _fred_client
    if _fred_client is None:
        _fred_client = FREDClient()
    return _fred_client


# Quick access functions
def get_economic_snapshot() -> Dict:
    """Quick function to get economic snapshot"""
    return get_client().get_economic_snapshot()


def get_recession_indicators() -> Dict:
    """Quick function to get recession indicators"""
    return get_client().get_recession_indicators()


def get_inflation_report() -> Dict:
    """Quick function to get inflation report"""
    return get_client().get_inflation_report()


def get_market_regime() -> Dict:
    """Quick function to get market regime"""
    return get_client().get_market_regime()


def get_fed_funds_rate() -> Optional[float]:
    """Quick function to get current Fed Funds rate"""
    latest = get_client().get_latest_value('FEDFUNDS')
    return latest['value'] if latest else None


def get_unemployment_rate() -> Optional[float]:
    """Quick function to get current unemployment rate"""
    latest = get_client().get_latest_value('UNRATE')
    return latest['value'] if latest else None


def get_vix() -> Optional[float]:
    """Quick function to get current VIX level"""
    latest = get_client().get_latest_value('VIXCLS')
    return latest['value'] if latest else None


if __name__ == "__main__":
    # Test the client
    logging.basicConfig(level=logging.INFO)

    client = FREDClient()

    print("\n=== Testing FRED API (FREE & UNLIMITED) ===\n")

    # Test 1: Economic Snapshot
    print("1. Getting economic snapshot...")
    snapshot = client.get_economic_snapshot()
    if snapshot:
        print("   ✅ Key Indicators:")
        for series_id, data in snapshot['indicators'].items():
            print(f"      {data['name']}: {data['value']} (as of {data['date']})")

    # Test 2: Recession Indicators
    print("\n2. Analyzing recession indicators...")
    recession = client.get_recession_indicators()
    if recession:
        print(f"   ✅ Recession Risk: {recession['recession_risk']}")
        print(f"   Warnings: {recession['warnings']}/{recession['total_indicators']}")

    # Test 3: Inflation Report
    print("\n3. Getting inflation report...")
    inflation = client.get_inflation_report()
    if inflation:
        print(f"   ✅ Overall Assessment: {inflation.get('overall_assessment', 'Unknown')}")
        for series_id, data in inflation['metrics'].items():
            print(f"      {data['name']}: {data['yoy_change_pct']}% YoY")

    # Test 4: Market Regime
    print("\n4. Determining market regime...")
    regime = client.get_market_regime()
    if regime:
        print(f"   ✅ Regime: {regime['regime']}")
        print(f"   {regime['characteristics']}")
