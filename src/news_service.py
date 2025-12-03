"""
News Service for fetching real-time market news from Finnhub and Polygon APIs.

Features:
- Fetches news from multiple sources (Finnhub and Polygon)
- Deduplicates news articles
- Caches results for 30 minutes
- Handles API errors gracefully
"""

import os
import requests
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from dataclasses import dataclass
import logging

# Try to load from dotenv
try:
    from dotenv import load_dotenv
    # Load from .env file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_file = os.path.join(os.path.dirname(current_dir), '.env')
    if os.path.exists(env_file):
        load_dotenv(env_file)
except ImportError:
    pass

# Try to import streamlit for caching
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Data class for news article"""
    symbol: str
    headline: str
    source: str
    url: str
    published_at: datetime
    summary: str
    sentiment: str  # 'positive', 'negative', 'neutral'


def cache_decorator(ttl=1800):
    """Decorator that uses st.cache_data if available, otherwise does nothing"""
    def decorator(func):
        if HAS_STREAMLIT:
            return st.cache_data(ttl=ttl)(func)
        return func
    return decorator


class NewsService:
    """Service for fetching and managing news from multiple sources"""

    def __init__(self) -> None:
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        self.cache_ttl = 1800  # 30 minutes

    def get_finnhub_news(self, symbol: str, days_back: int = 7) -> List[NewsArticle]:
        """
        Fetch news from Finnhub API

        Args:
            symbol: Stock ticker symbol
            days_back: Number of days to look back for news

        Returns:
            List of NewsArticle objects
        """
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            'symbol': symbol,
            'from': (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
            'to': datetime.now().strftime('%Y-%m-%d'),
            'token': self.finnhub_key
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data[:10]:  # Limit to 10 articles
                try:
                    # Finnhub returns unix timestamp, convert to UTC datetime
                    timestamp = item.get('datetime', 0)
                    published_at = datetime.fromtimestamp(timestamp, tz=timezone.utc)

                    articles.append(NewsArticle(
                        symbol=symbol,
                        headline=item.get('headline', 'No headline'),
                        source=item.get('source', 'Unknown'),
                        url=item.get('url', '#'),
                        published_at=published_at,
                        summary=item.get('summary', ''),
                        sentiment='neutral'  # Finnhub doesn't provide sentiment in free tier
                    ))
                except Exception as e:
                    logger.warning(f"Error parsing Finnhub article for {symbol}: {e}")
                    continue

            logger.info(f"Successfully fetched {len(articles)} articles from Finnhub for {symbol}")
            return articles
        except requests.exceptions.Timeout:
            logger.error(f"Finnhub API timeout for {symbol}")
            return []
        except requests.exceptions.HTTPError as e:
            logger.error(f"Finnhub API HTTP error for {symbol}: {e}")
            return []
        except Exception as e:
            logger.error(f"Finnhub API error for {symbol}: {e}")
            return []

    def get_polygon_news(self, symbol: str, limit: int = 10) -> List[NewsArticle]:
        """
        Fetch news from Polygon API

        Args:
            symbol: Stock ticker symbol
            limit: Maximum number of articles to return

        Returns:
            List of NewsArticle objects
        """
        url = "https://api.polygon.io/v2/reference/news"
        params = {
            'ticker': symbol,
            'limit': limit,
            'apiKey': self.polygon_key
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            articles = []
            for item in data.get('results', []):
                try:
                    # Parse the published_utc timestamp (ISO 8601 format)
                    pub_time = item.get('published_utc', '')
                    if pub_time:
                        # Handle ISO format with Z or +00:00
                        pub_time = pub_time.replace('Z', '+00:00')
                        published_at = datetime.fromisoformat(pub_time)
                    else:
                        published_at = datetime.now(tz=timezone.utc)

                    articles.append(NewsArticle(
                        symbol=symbol,
                        headline=item.get('title', 'No headline'),
                        source=item.get('publisher', {}).get('name', 'Unknown') if isinstance(item.get('publisher'), dict) else 'Unknown',
                        url=item.get('article_url', '#'),
                        published_at=published_at,
                        summary=item.get('description', ''),
                        sentiment='neutral'
                    ))
                except Exception as e:
                    logger.warning(f"Error parsing Polygon article for {symbol}: {e}")
                    continue

            logger.info(f"Successfully fetched {len(articles)} articles from Polygon for {symbol}")
            return articles
        except requests.exceptions.Timeout:
            logger.error(f"Polygon API timeout for {symbol}")
            return []
        except requests.exceptions.HTTPError as e:
            logger.error(f"Polygon API HTTP error for {symbol}: {e}")
            return []
        except Exception as e:
            logger.error(f"Polygon API error for {symbol}: {e}")
            return []

    def get_combined_news(self, symbol: str) -> List[NewsArticle]:
        """
        Get news from both sources and combine with deduplication

        Args:
            symbol: Stock ticker symbol

        Returns:
            List of deduplicated NewsArticle objects, sorted by date (newest first)
        """
        finnhub_news = self.get_finnhub_news(symbol)
        polygon_news = self.get_polygon_news(symbol)

        # Combine and deduplicate by headline
        all_news = finnhub_news + polygon_news
        seen_headlines = set()
        unique_news = []

        for article in all_news:
            # Normalize headline for comparison
            normalized_headline = article.headline.lower().strip()
            if normalized_headline not in seen_headlines:
                seen_headlines.add(normalized_headline)
                unique_news.append(article)

        # Sort by date (newest first)
        # All datetimes should be timezone-aware now
        unique_news.sort(key=lambda x: x.published_at, reverse=True)
        return unique_news[:15]  # Return top 15
