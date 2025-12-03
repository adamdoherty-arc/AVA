"""
Data Source Integrations

Manages integration with free data sources for multi-sector predictions:
- Reddit API (sentiment analysis)
- FRED API (economic indicators)
- CoinGecko API (crypto prices)
- yfinance (stock market data)

Author: Python Pro
Created: 2025-11-09
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import time

import requests
import praw  # Reddit API

logger = logging.getLogger(__name__)


@dataclass
class DataSourceResult:
    """Result from a data source query"""
    source: str
    data: Dict[str, Any]
    timestamp: datetime
    success: bool
    error: Optional[str] = None


class DataSourceManager:
    """Manages all external data source integrations"""

    def __init__(self) -> None:
        """Initialize data source manager"""
        self.reddit_client = self._init_reddit()
        self.fred_api_key = os.getenv('FRED_API_KEY')
        self.cache: Dict[str, DataSourceResult] = {}
        self.cache_ttl = 3600  # 1 hour cache

    def _init_reddit(self) -> Optional[praw.Reddit]:
        """Initialize Reddit API client (no auth needed for read-only)"""
        try:
            reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID', 'anonymous'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET', 'anonymous'),
                user_agent='PredictionAgent/1.0 by Python-Pro'
            )
            return reddit
        except Exception as e:
            logger.warning(f"Reddit client initialization failed: {e}")
            return None

    def _get_cache_key(self, source: str, params: Dict) -> str:
        """Generate cache key from source and parameters"""
        param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{source}_{param_str}"

    def _check_cache(self, cache_key: str) -> Optional[DataSourceResult]:
        """Check if cached result is still valid"""
        if cache_key not in self.cache:
            return None

        result = self.cache[cache_key]
        age = (datetime.now() - result.timestamp).total_seconds()

        if age < self.cache_ttl:
            logger.debug(f"Cache hit for {cache_key} (age: {age:.0f}s)")
            return result

        # Expired, remove from cache
        del self.cache[cache_key]
        return None

    def _set_cache(self, cache_key: str, result: DataSourceResult):
        """Store result in cache"""
        self.cache[cache_key] = result

    # ========================================================================
    # REDDIT DATA SOURCE
    # ========================================================================

    def get_reddit_sentiment(self, subreddit: str, keywords: List[str],
                            limit: int = 100) -> DataSourceResult:
        """
        Get sentiment data from Reddit

        Args:
            subreddit: Subreddit name (e.g., 'nfl', 'politics')
            keywords: Keywords to search for
            limit: Number of posts to analyze

        Returns:
            DataSourceResult with sentiment metrics
        """
        cache_key = self._get_cache_key('reddit', {
            'subreddit': subreddit,
            'keywords': '_'.join(keywords),
            'limit': limit
        })

        # Check cache
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        if not self.reddit_client:
            result = DataSourceResult(
                source='reddit',
                data={},
                timestamp=datetime.now(),
                success=False,
                error="Reddit client not initialized"
            )
            return result

        try:
            # Search for posts with keywords
            posts = []
            keyword_query = ' OR '.join(keywords)

            subreddit_obj = self.reddit_client.subreddit(subreddit)
            for post in subreddit_obj.search(keyword_query, limit=limit, sort='new'):
                posts.append({
                    'title': post.title,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created': datetime.fromtimestamp(post.created_utc),
                    'upvote_ratio': post.upvote_ratio
                })

            # Calculate sentiment metrics
            if posts:
                avg_score = sum(p['score'] for p in posts) / len(posts)
                avg_comments = sum(p['num_comments'] for p in posts) / len(posts)
                avg_upvote_ratio = sum(p['upvote_ratio'] for p in posts) / len(posts)

                # Sentiment score (0-100): higher is more positive
                sentiment_score = min(avg_upvote_ratio * 100, 100)
                engagement = min((avg_score + avg_comments) / 10, 100)

                data = {
                    'sentiment_score': sentiment_score,
                    'engagement': engagement,
                    'post_count': len(posts),
                    'avg_score': avg_score,
                    'avg_comments': avg_comments,
                    'avg_upvote_ratio': avg_upvote_ratio
                }
            else:
                data = {
                    'sentiment_score': 50.0,  # Neutral
                    'engagement': 0.0,
                    'post_count': 0
                }

            result = DataSourceResult(
                source='reddit',
                data=data,
                timestamp=datetime.now(),
                success=True
            )

        except Exception as e:
            logger.error(f"Reddit API error: {e}")
            result = DataSourceResult(
                source='reddit',
                data={'sentiment_score': 50.0, 'engagement': 0.0},
                timestamp=datetime.now(),
                success=False,
                error=str(e)
            )

        self._set_cache(cache_key, result)
        return result

    # ========================================================================
    # FRED (ECONOMIC DATA)
    # ========================================================================

    def get_fred_indicator(self, series_id: str,
                          lookback_days: int = 90) -> DataSourceResult:
        """
        Get economic indicator from FRED API

        Args:
            series_id: FRED series ID (e.g., 'GDP', 'UNRATE', 'CPIAUCSL')
            lookback_days: Days of historical data

        Returns:
            DataSourceResult with indicator data
        """
        cache_key = self._get_cache_key('fred', {
            'series_id': series_id,
            'lookback': lookback_days
        })

        # Check cache
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        if not self.fred_api_key:
            logger.warning("FRED API key not set. Set FRED_API_KEY environment variable.")
            result = DataSourceResult(
                source='fred',
                data={},
                timestamp=datetime.now(),
                success=False,
                error="FRED API key not configured"
            )
            return result

        try:
            # FRED API endpoint
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')

            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'observation_start': start_date,
                'observation_end': end_date
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            observations = response.json().get('observations', [])

            if observations:
                # Get latest value and calculate trend
                values = [float(obs['value']) for obs in observations if obs['value'] != '.']

                if values:
                    latest_value = values[-1]
                    first_value = values[0]
                    percent_change = ((latest_value - first_value) / first_value * 100) if first_value != 0 else 0

                    data = {
                        'series_id': series_id,
                        'latest_value': latest_value,
                        'percent_change': percent_change,
                        'observations': len(values),
                        'trend': 'up' if percent_change > 0 else 'down' if percent_change < 0 else 'flat'
                    }
                else:
                    data = {}
            else:
                data = {}

            result = DataSourceResult(
                source='fred',
                data=data,
                timestamp=datetime.now(),
                success=bool(data)
            )

        except Exception as e:
            logger.error(f"FRED API error: {e}")
            result = DataSourceResult(
                source='fred',
                data={},
                timestamp=datetime.now(),
                success=False,
                error=str(e)
            )

        self._set_cache(cache_key, result)
        return result

    # ========================================================================
    # COINGECKO (CRYPTO DATA)
    # ========================================================================

    def get_crypto_data(self, coin_id: str, days: int = 30) -> DataSourceResult:
        """
        Get cryptocurrency data from CoinGecko API (free, no key required)

        Args:
            coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum')
            days: Days of historical data

        Returns:
            DataSourceResult with crypto metrics
        """
        cache_key = self._get_cache_key('coingecko', {
            'coin_id': coin_id,
            'days': days
        })

        # Check cache
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            # Current price and market data
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            params = {
                'localization': 'false',
                'tickers': 'false',
                'community_data': 'false',
                'developer_data': 'false'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            coin_data = response.json()

            # Extract key metrics
            market_data = coin_data.get('market_data', {})

            data = {
                'coin_id': coin_id,
                'current_price': market_data.get('current_price', {}).get('usd', 0),
                'market_cap': market_data.get('market_cap', {}).get('usd', 0),
                'total_volume': market_data.get('total_volume', {}).get('usd', 0),
                'price_change_24h': market_data.get('price_change_percentage_24h', 0),
                'price_change_7d': market_data.get('price_change_percentage_7d', 0),
                'price_change_30d': market_data.get('price_change_percentage_30d', 0),
                'market_cap_rank': market_data.get('market_cap_rank', 0),
                'ath': market_data.get('ath', {}).get('usd', 0),
                'atl': market_data.get('atl', {}).get('usd', 0)
            }

            result = DataSourceResult(
                source='coingecko',
                data=data,
                timestamp=datetime.now(),
                success=True
            )

            # Rate limiting: CoinGecko free tier allows ~10-50 calls/min
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"CoinGecko API error: {e}")
            result = DataSourceResult(
                source='coingecko',
                data={},
                timestamp=datetime.now(),
                success=False,
                error=str(e)
            )

        self._set_cache(cache_key, result)
        return result

    # ========================================================================
    # YFINANCE (STOCK DATA)
    # ========================================================================

    def get_market_data(self, symbol: str, period: str = '1mo') -> DataSourceResult:
        """
        Get market data from yfinance (free, no key required)

        Args:
            symbol: Stock symbol (e.g., 'SPY', '^VIX', '^DJI')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y')

        Returns:
            DataSourceResult with market metrics
        """
        cache_key = self._get_cache_key('yfinance', {
            'symbol': symbol,
            'period': period
        })

        # Check cache
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            if hist.empty:
                raise ValueError(f"No data for symbol: {symbol}")

            # Calculate metrics
            latest_price = float(hist['Close'].iloc[-1])
            first_price = float(hist['Close'].iloc[0])
            percent_change = ((latest_price - first_price) / first_price * 100)

            # Volatility (standard deviation of returns)
            returns = hist['Close'].pct_change().dropna()
            volatility = float(returns.std() * 100)

            # Volume trend
            avg_volume = float(hist['Volume'].mean())
            latest_volume = float(hist['Volume'].iloc[-1])
            volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1.0

            data = {
                'symbol': symbol,
                'latest_price': latest_price,
                'percent_change': percent_change,
                'volatility': volatility,
                'avg_volume': avg_volume,
                'latest_volume': latest_volume,
                'volume_ratio': volume_ratio,
                'high': float(hist['High'].max()),
                'low': float(hist['Low'].min())
            }

            result = DataSourceResult(
                source='yfinance',
                data=data,
                timestamp=datetime.now(),
                success=True
            )

        except Exception as e:
            logger.error(f"yfinance error: {e}")
            result = DataSourceResult(
                source='yfinance',
                data={},
                timestamp=datetime.now(),
                success=False,
                error=str(e)
            )

        self._set_cache(cache_key, result)
        return result

    # ========================================================================
    # MULTI-SOURCE AGGREGATION
    # ========================================================================

    def get_sector_data(self, sector: str, context: Dict[str, Any]) -> Dict[str, DataSourceResult]:
        """
        Get all relevant data for a sector

        Args:
            sector: Sector name ('sports', 'politics', 'economics', 'crypto')
            context: Context dictionary with market-specific info

        Returns:
            Dictionary of data source results
        """
        results: Dict[str, DataSourceResult] = {}

        if sector == 'sports':
            # Reddit sentiment for sports
            sport_type = context.get('sport', 'nfl')
            results['reddit'] = self.get_reddit_sentiment(
                subreddit=sport_type,
                keywords=context.get('keywords', [sport_type])
            )

        elif sector == 'politics':
            # Reddit political sentiment
            results['reddit'] = self.get_reddit_sentiment(
                subreddit='politics',
                keywords=context.get('keywords', ['election'])
            )

        elif sector == 'economics':
            # FRED indicators + market data
            indicators = context.get('indicators', ['UNRATE', 'CPIAUCSL'])
            for indicator in indicators:
                results[f'fred_{indicator}'] = self.get_fred_indicator(indicator)

            # Market indices
            indices = context.get('indices', ['SPY', '^VIX'])
            for index in indices:
                results[f'market_{index}'] = self.get_market_data(index)

        elif sector == 'crypto':
            # Crypto data
            coins = context.get('coins', ['bitcoin'])
            for coin in coins:
                results[f'crypto_{coin}'] = self.get_crypto_data(coin)

            # Crypto sentiment
            results['reddit'] = self.get_reddit_sentiment(
                subreddit='cryptocurrency',
                keywords=context.get('keywords', ['bitcoin'])
            )

        return results

    def clear_cache(self) -> None:
        """Clear all cached data"""
        self.cache.clear()
        logger.info("Data source cache cleared")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    manager = DataSourceManager()

    print("\n" + "="*80)
    print("DATA SOURCE MANAGER TEST")
    print("="*80)

    # Test Reddit sentiment
    print("\n1. Testing Reddit Sentiment (r/nfl)...")
    result = manager.get_reddit_sentiment('nfl', ['chiefs', 'bills'], limit=50)
    print(f"   Success: {result.success}")
    if result.success:
        print(f"   Sentiment Score: {result.data.get('sentiment_score', 0):.1f}/100")
        print(f"   Post Count: {result.data.get('post_count', 0)}")

    # Test CoinGecko
    print("\n2. Testing CoinGecko (Bitcoin)...")
    result = manager.get_crypto_data('bitcoin')
    print(f"   Success: {result.success}")
    if result.success:
        print(f"   Current Price: ${result.data.get('current_price', 0):,.2f}")
        print(f"   24h Change: {result.data.get('price_change_24h', 0):.2f}%")

    # Test yfinance
    print("\n3. Testing yfinance (SPY)...")
    result = manager.get_market_data('SPY')
    print(f"   Success: {result.success}")
    if result.success:
        print(f"   Latest Price: ${result.data.get('latest_price', 0):.2f}")
        print(f"   Volatility: {result.data.get('volatility', 0):.2f}%")

    print("\n" + "="*80)
