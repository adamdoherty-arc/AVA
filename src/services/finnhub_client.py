"""
Finnhub API Client (FREE Tier)
================================

FREE access to comprehensive market data with generous rate limits.

Features (FREE Tier):
- Real-time stock quotes
- Company news & press releases
- Financial statements & metrics
- Insider transactions
- Market news
- Stock symbols & company profiles
- Earnings calendar

Free Tier: 60 API calls per minute (very generous!)

API Key: Get FREE at https://finnhub.io/register

OPTIMIZATIONS APPLIED:
1. Minimal rate limiting (tracks per-minute calls, no blocking sleep)
2. Redis/In-Memory caching with proper TTLs
3. Async-compatible design
4. Circuit breaker pattern for resilience

Author: Magnus Trading Platform
Created: 2025-11-21
Updated: 2025-11-29 - Added circuit breaker pattern
"""

import os
import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern for API resilience.
    Prevents cascading failures by temporarily blocking requests to a failing service.
    """
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    _failure_count: int = field(default=0, init=False, repr=False)
    _last_failure_time: Optional[float] = field(default=None, init=False, repr=False)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False, repr=False)

    def record_success(self) -> None:
        """Record a successful request - reset circuit"""
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed request - may trip circuit"""
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(f"Finnhub circuit breaker OPEN after {self._failure_count} failures")

    def can_execute(self) -> bool:
        """Check if request can proceed"""
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.OPEN:
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    logger.info("Finnhub circuit breaker HALF_OPEN - testing recovery")
                    return True
            return False
        return True  # HALF_OPEN allows one test request

    @property
    def state(self) -> CircuitState:
        return self._state


class FinnhubClient:
    """
    Client for Finnhub FREE API with optimized rate limiting and circuit breaker.

    Performance Improvements:
    - Non-blocking rate limiting (tracks per-minute calls)
    - Redis/In-memory caching for frequently accessed data
    - 60 calls/minute limit respected without blocking sleeps
    - Circuit breaker pattern for resilience (trips after 5 failures, 60s recovery)
    """

    BASE_URL = "https://finnhub.io/api/v1"

    # Demo API key (get your own free key)
    DEFAULT_API_KEY = "demo"

    # Cache TTLs
    CACHE_TTL_QUOTE = 30          # 30 seconds for quotes
    CACHE_TTL_NEWS = 300          # 5 minutes for news
    CACHE_TTL_PROFILE = 3600      # 1 hour for company profiles
    CACHE_TTL_METRICS = 1800      # 30 minutes for metrics

    # Rate limit: 60 calls per minute
    RATE_LIMIT_PER_MINUTE = 60

    def __init__(self, api_key: Optional[str] = None, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        """
        Initialize Finnhub client with circuit breaker.

        Args:
            api_key: Finnhub API key (get FREE at finnhub.io)
                    Falls back to env var FINNHUB_API_KEY
            failure_threshold: Number of failures before circuit opens (default: 5)
            recovery_timeout: Seconds before circuit attempts recovery (default: 60)
        """
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY', self.DEFAULT_API_KEY)
        self.session = requests.Session()
        self.call_count = 0
        self.last_call_time = None

        # Rate limiting: track calls in current minute
        self._minute_calls = 0
        self._minute_reset = datetime.now()

        # Circuit breaker for resilience
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )

        # Try to use distributed cache
        try:
            from backend.infrastructure.cache import get_cache
            self._cache = get_cache()
            self._use_cache = True
        except ImportError:
            self._use_cache = False
            self._local_cache: Dict[str, tuple] = {}

        logger.info(f"Finnhub client initialized with circuit breaker (API key: {self.api_key[:8]}...)")

    @property
    def circuit_state(self) -> str:
        """Get current circuit breaker state"""
        return self._circuit_breaker.state.value

    def _check_rate_limit(self) -> bool:
        """
        Check if we're within rate limits for this minute.

        Returns:
            True if we can make a request, False if at limit.
        """
        now = datetime.now()
        # Reset counter if we're in a new minute
        if (now - self._minute_reset).total_seconds() >= 60:
            self._minute_reset = now
            self._minute_calls = 0
        return self._minute_calls < self.RATE_LIMIT_PER_MINUTE

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value (sync-compatible)."""
        if self._use_cache:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    return None  # Can't use async cache in sync context
                return loop.run_until_complete(self._cache.get(key))
            except Exception:
                return None
        else:
            # Use local cache with TTL check
            if key in self._local_cache:
                value, expiry = self._local_cache[key]
                if datetime.now() < expiry:
                    return value
                del self._local_cache[key]
            return None

    def _set_cache(self, key: str, value: Any, ttl: int = 300) -> None:
        """Set cached value (sync-compatible)."""
        if self._use_cache:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    loop.run_until_complete(self._cache.set(key, value, ttl))
            except Exception:
                pass
        else:
            expiry = datetime.now() + timedelta(seconds=ttl)
            self._local_cache[key] = (value, expiry)

    def _make_request(
        self, endpoint: str, params: Optional[Dict] = None
    ) -> Optional[Any]:
        """
        Make API request with smart rate limiting and circuit breaker protection.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            JSON response or None if blocked/failed
        """
        if params is None:
            params = {}

        # Check circuit breaker first
        if not self._circuit_breaker.can_execute():
            logger.warning(f"Finnhub circuit breaker OPEN - request blocked: {endpoint}")
            return None

        # Check rate limit - log warning if approaching limit
        if not self._check_rate_limit():
            logger.warning("Finnhub rate limit reached (60/min)")
            return None

        # Add API key
        params['token'] = self.api_key

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            self.last_call_time = time.time()
            self.call_count += 1
            self._minute_calls += 1

            data = response.json()

            # Check for API-level errors
            if isinstance(data, dict) and 'error' in data:
                logger.error(f"Finnhub API error: {data['error']}")
                self._circuit_breaker.record_failure()
                return None

            # Success - reset circuit breaker
            self._circuit_breaker.record_success()
            logger.debug(f"Finnhub API call #{self.call_count} successful")
            return data

        except requests.exceptions.RequestException as e:
            self._circuit_breaker.record_failure()
            logger.error(f"Finnhub request failed ({self._circuit_breaker.state.value}): {e}")
            return None

    # ========================================================================
    # STOCK QUOTES & PRICES
    # ========================================================================

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time quote for a symbol (FREE).

        Uses caching with 30-second TTL.

        Args:
            symbol: Stock ticker

        Returns:
            Quote with current price, change, high, low, etc.
        """
        # Check cache first
        cache_key = f"finnhub:quote:{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        data = self._make_request('quote', {'symbol': symbol})

        if not data:
            return None

        ts = data.get('t')
        timestamp = datetime.fromtimestamp(ts).isoformat() if ts else None
        result = {
            'symbol': symbol,
            'current_price': data.get('c', 0),
            'change': data.get('d', 0),
            'percent_change': data.get('dp', 0),
            'high': data.get('h', 0),
            'low': data.get('l', 0),
            'open': data.get('o', 0),
            'previous_close': data.get('pc', 0),
            'timestamp': timestamp
        }

        # Cache the result
        self._set_cache(cache_key, result, self.CACHE_TTL_QUOTE)
        return result

    def get_candles(
        self,
        symbol: str,
        resolution: str = 'D',
        days_back: int = 30
    ) -> Optional[List[Dict]]:
        """
        Get historical price candles (FREE).

        Args:
            symbol: Stock ticker
            resolution: '1', '5', '15', '30', '60', 'D', 'W', 'M'
            days_back: How many days of history

        Returns:
            List of OHLCV candles
        """
        end_time = int(datetime.now().timestamp())
        start_time = int((datetime.now() - timedelta(days=days_back)).timestamp())

        data = self._make_request('stock/candle', {
            'symbol': symbol,
            'resolution': resolution,
            'from': start_time,
            'to': end_time
        })

        if not data or data.get('s') != 'ok':
            return None

        # Convert to list of candles
        candles = []
        for i in range(len(data.get('t', []))):
            candles.append({
                'timestamp': datetime.fromtimestamp(data['t'][i]).isoformat(),
                'open': data['o'][i],
                'high': data['h'][i],
                'low': data['l'][i],
                'close': data['c'][i],
                'volume': data['v'][i]
            })

        return candles

    # ========================================================================
    # COMPANY DATA
    # ========================================================================

    def get_company_profile(self, symbol: str) -> Optional[Dict]:
        """
        Get company profile and metadata (FREE).

        Args:
            symbol: Stock ticker

        Returns:
            Company info including name, industry, market cap, etc.
        """
        data = self._make_request('stock/profile2', {'symbol': symbol})

        if not data:
            return None

        return {
            'symbol': symbol,
            'name': data.get('name', ''),
            'ticker': data.get('ticker', symbol),
            'exchange': data.get('exchange', ''),
            'industry': data.get('finnhubIndustry', ''),
            'market_cap': data.get('marketCapitalization', 0),
            'share_outstanding': data.get('shareOutstanding', 0),
            'ipo_date': data.get('ipo', ''),
            'logo': data.get('logo', ''),
            'phone': data.get('phone', ''),
            'weburl': data.get('weburl', ''),
            'country': data.get('country', ''),
            'currency': data.get('currency', '')
        }

    def get_company_metrics(self, symbol: str) -> Optional[Dict]:
        """
        Get company financial metrics (FREE).

        Args:
            symbol: Stock ticker

        Returns:
            Key metrics like P/E, beta, 52-week high/low, etc.
        """
        data = self._make_request('stock/metric', {'symbol': symbol, 'metric': 'all'})

        if not data or 'metric' not in data:
            return None

        metrics = data['metric']

        return {
            'symbol': symbol,
            '52_week_high': metrics.get('52WeekHigh', 0),
            '52_week_low': metrics.get('52WeekLow', 0),
            'beta': metrics.get('beta', 0),
            'pe_ratio': metrics.get('peBasicExclExtraTTM', 0),
            'price_to_book': metrics.get('pbAnnual', 0),
            'price_to_sales': metrics.get('psTTM', 0),
            'dividend_yield': metrics.get('dividendYieldIndicatedAnnual', 0),
            'roe': metrics.get('roeTTM', 0),
            'roa': metrics.get('roaTTM', 0),
            'profit_margin': metrics.get('netProfitMarginTTM', 0),
            'operating_margin': metrics.get('operatingMarginTTM', 0),
            'revenue_per_share': metrics.get('revenuePerShareTTM', 0),
            'eps': metrics.get('epsExclExtraItemsTTM', 0)
        }

    # ========================================================================
    # NEWS & PRESS RELEASES
    # ========================================================================

    def get_company_news(
        self,
        symbol: str,
        days_back: int = 7
    ) -> Optional[List[Dict]]:
        """
        Get company-specific news (FREE).

        Args:
            symbol: Stock ticker
            days_back: Days to look back (max 365)

        Returns:
            List of news articles with headline, summary, source, URL
        """
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')

        data = self._make_request('company-news', {
            'symbol': symbol,
            'from': from_date,
            'to': to_date
        })

        if not data:
            return None

        articles = []
        for article in data:
            articles.append({
                'headline': article.get('headline', ''),
                'summary': article.get('summary', ''),
                'source': article.get('source', ''),
                'url': article.get('url', ''),
                'datetime': datetime.fromtimestamp(article.get('datetime', 0)).isoformat() if article.get('datetime') else None,
                'category': article.get('category', ''),
                'image': article.get('image', '')
            })

        return articles

    def get_market_news(self, category: str = 'general') -> Optional[List[Dict]]:
        """
        Get general market news (FREE).

        Args:
            category: 'general', 'forex', 'crypto', 'merger'

        Returns:
            List of market news articles
        """
        data = self._make_request('news', {'category': category})

        if not data:
            return None

        articles = []
        for article in data[:20]:  # Limit to 20 articles
            articles.append({
                'headline': article.get('headline', ''),
                'summary': article.get('summary', ''),
                'source': article.get('source', ''),
                'url': article.get('url', ''),
                'datetime': datetime.fromtimestamp(article.get('datetime', 0)).isoformat() if article.get('datetime') else None,
                'category': article.get('category', ''),
                'image': article.get('image', '')
            })

        return articles

    # ========================================================================
    # INSIDER TRANSACTIONS
    # ========================================================================

    def get_insider_transactions(
        self,
        symbol: str,
        months_back: int = 3
    ) -> Optional[List[Dict]]:
        """
        Get insider trading transactions (FREE).

        Args:
            symbol: Stock ticker
            months_back: Months to look back

        Returns:
            List of insider transactions with name, shares, price, etc.
        """
        from_date = (datetime.now() - timedelta(days=months_back*30)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')

        data = self._make_request('stock/insider-transactions', {
            'symbol': symbol,
            'from': from_date,
            'to': to_date
        })

        if not data or 'data' not in data:
            return None

        transactions = []
        for txn in data['data']:
            transactions.append({
                'name': txn.get('name', ''),
                'share': txn.get('share', 0),
                'change': txn.get('change', 0),
                'filing_date': txn.get('filingDate', ''),
                'transaction_date': txn.get('transactionDate', ''),
                'transaction_code': txn.get('transactionCode', ''),
                'transaction_price': txn.get('transactionPrice', 0)
            })

        return transactions

    # ========================================================================
    # EARNINGS
    # ========================================================================

    def get_earnings_calendar(
        self,
        symbol: Optional[str] = None,
        days_ahead: int = 30
    ) -> Optional[List[Dict]]:
        """
        Get earnings calendar (FREE).

        Args:
            symbol: Stock ticker (optional - if None, returns all)
            days_ahead: Days to look ahead

        Returns:
            List of upcoming earnings with date, EPS estimate, etc.
        """
        from_date = datetime.now().strftime('%Y-%m-%d')
        to_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')

        params = {
            'from': from_date,
            'to': to_date
        }

        if symbol:
            params['symbol'] = symbol

        data = self._make_request('calendar/earnings', params)

        if not data or 'earningsCalendar' not in data:
            return None

        earnings = []
        for item in data['earningsCalendar']:
            earnings.append({
                'symbol': item.get('symbol', ''),
                'date': item.get('date', ''),
                'eps_estimate': item.get('epsEstimate', 0),
                'eps_actual': item.get('epsActual', 0),
                'revenue_estimate': item.get('revenueEstimate', 0),
                'revenue_actual': item.get('revenueActual', 0)
            })

        return earnings

    # ========================================================================
    # RECOMMENDATIONS
    # ========================================================================

    def get_recommendations(self, symbol: str) -> Optional[List[Dict]]:
        """
        Get analyst recommendations (FREE).

        Args:
            symbol: Stock ticker

        Returns:
            Recent analyst recommendations
        """
        data = self._make_request('stock/recommendation', {'symbol': symbol})

        if not data:
            return None

        recommendations = []
        for rec in data[:12]:  # Last 12 months
            recommendations.append({
                'symbol': symbol,
                'period': rec.get('period', ''),
                'strong_buy': rec.get('strongBuy', 0),
                'buy': rec.get('buy', 0),
                'hold': rec.get('hold', 0),
                'sell': rec.get('sell', 0),
                'strong_sell': rec.get('strongSell', 0)
            })

        return recommendations

    def get_price_target(self, symbol: str) -> Optional[Dict]:
        """
        Get analyst price targets (FREE).

        Args:
            symbol: Stock ticker

        Returns:
            Price target consensus (high, low, average, median)
        """
        data = self._make_request('stock/price-target', {'symbol': symbol})

        if not data:
            return None

        return {
            'symbol': symbol,
            'target_high': data.get('targetHigh', 0),
            'target_low': data.get('targetLow', 0),
            'target_mean': data.get('targetMean', 0),
            'target_median': data.get('targetMedian', 0),
            'last_updated': data.get('lastUpdated', '')
        }

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def get_comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive analysis for a stock (combines multiple endpoints).

        Args:
            symbol: Stock ticker

        Returns:
            Complete analysis including quote, profile, news, insiders, etc.
        """
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat()
        }

        # Quote
        quote = self.get_quote(symbol)
        if quote:
            analysis['quote'] = quote

        # Company profile
        profile = self.get_company_profile(symbol)
        if profile:
            analysis['profile'] = profile

        # Metrics
        metrics = self.get_company_metrics(symbol)
        if metrics:
            analysis['metrics'] = metrics

        # Recent news
        news = self.get_company_news(symbol, days_back=7)
        if news:
            analysis['recent_news'] = news[:5]  # Top 5 articles

        # Insider activity
        insiders = self.get_insider_transactions(symbol, months_back=3)
        if insiders:
            analysis['insider_transactions'] = insiders[:10]  # Top 10

        # Analyst recommendations
        recs = self.get_recommendations(symbol)
        if recs:
            analysis['analyst_recommendations'] = recs[0] if recs else None

        # Price target
        target = self.get_price_target(symbol)
        if target:
            analysis['price_target'] = target

        return analysis

    def get_usage_stats(self) -> Dict:
        """Get API usage statistics including circuit breaker state"""
        return {
            'total_calls': self.call_count,
            'calls_per_minute_limit': 60,
            'last_call': self.last_call_time,
            'api_key': f"{self.api_key[:8]}...",
            'free_tier': True,
            'circuit_breaker_state': self.circuit_state
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_finnhub_client = None

def get_client() -> FinnhubClient:
    """Get singleton Finnhub client instance"""
    global _finnhub_client
    if _finnhub_client is None:
        _finnhub_client = FinnhubClient()
    return _finnhub_client


# Quick access functions
def get_quote(symbol: str) -> Optional[Dict]:
    """Quick function to get a quote"""
    return get_client().get_quote(symbol)


def get_company_news(symbol: str, days_back: int = 7) -> Optional[List[Dict]]:
    """Quick function to get company news"""
    return get_client().get_company_news(symbol, days_back)


def get_company_profile(symbol: str) -> Optional[Dict]:
    """Quick function to get company profile"""
    return get_client().get_company_profile(symbol)


def get_insider_transactions(symbol: str) -> Optional[List[Dict]]:
    """Quick function to get insider transactions"""
    return get_client().get_insider_transactions(symbol)


def get_comprehensive_analysis(symbol: str) -> Dict:
    """Quick function to get comprehensive analysis"""
    return get_client().get_comprehensive_analysis(symbol)


if __name__ == "__main__":
    # Test the client
    logging.basicConfig(level=logging.INFO)

    client = FinnhubClient()

    print("\n=== Testing Finnhub API (FREE - 60 calls/min) ===\n")

    # Test 1: Get quote
    print("1. Getting AAPL quote...")
    quote = client.get_quote('AAPL')
    if quote:
        print(f"   ✅ AAPL: ${quote['current_price']} ({quote['percent_change']:+.2f}%)")

    # Test 2: Get company profile
    print("\n2. Getting AAPL company profile...")
    profile = client.get_company_profile('AAPL')
    if profile:
        print(f"   ✅ {profile['name']}")
        print(f"   Market Cap: ${profile['market_cap']}B")
        print(f"   Industry: {profile['industry']}")

    # Test 3: Get news
    print("\n3. Getting AAPL news...")
    news = client.get_company_news('AAPL', days_back=3)
    if news:
        print(f"   ✅ Found {len(news)} articles")
        print(f"   Latest: {news[0]['headline']}")

    # Test 4: Get insider transactions
    print("\n4. Getting AAPL insider transactions...")
    insiders = client.get_insider_transactions('AAPL', months_back=1)
    if insiders:
        print(f"   ✅ Found {len(insiders)} insider transactions")

    print(f"\n=== Usage Stats ===")
    stats = client.get_usage_stats()
    print(f"API calls made: {stats['total_calls']}")
    print(f"Rate limit: {stats['calls_per_minute_limit']} calls/min")
