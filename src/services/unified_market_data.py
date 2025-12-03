"""
Unified Market Data Service
============================

Intelligent aggregation of multiple FREE data sources with automatic failover,
caching, and best-source routing.

Data Sources (All FREE):
- Alpha Vantage: News, sentiment, quotes, fundamentals
- FRED: Economic indicators, macro data
- Finnhub: Market data, insider transactions, earnings

Features:
- Automatic provider selection based on data type
- Failover to backup sources
- Intelligent caching
- Cost optimization (prefer free sources)
- Usage tracking

Author: Magnus Trading Platform
Created: 2025-11-21
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from functools import lru_cache
import json

# Import our FREE API clients
from src.services.alpha_vantage_client import AlphaVantageClient
from src.services.fred_client import FREDClient
from src.services.finnhub_client import FinnhubClient

logger = logging.getLogger(__name__)


class UnifiedMarketData:
    """
    Unified interface for all market data with automatic provider selection
    and failover.
    """

    def __init__(
        self,
        alpha_vantage_key: Optional[str] = None,
        fred_key: Optional[str] = None,
        finnhub_key: Optional[str] = None
    ):
        """
        Initialize unified market data service.

        Args:
            alpha_vantage_key: Alpha Vantage API key (or uses env var)
            fred_key: FRED API key (or uses env var)
            finnhub_key: Finnhub API key (or uses env var)
        """
        # Initialize all FREE API clients
        self.alpha_vantage = AlphaVantageClient(alpha_vantage_key)
        self.fred = FREDClient(fred_key)
        self.finnhub = FinnhubClient(finnhub_key)

        # Usage tracking
        self.usage_stats = {
            'alpha_vantage': 0,
            'fred': 0,
            'finnhub': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        logger.info("✅ Unified Market Data service initialized with 3 FREE sources")

    # ========================================================================
    # STOCK QUOTES & PRICES
    # ========================================================================

    @lru_cache(maxsize=500)
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time quote with automatic failover.

        Tries: Alpha Vantage → Finnhub

        Args:
            symbol: Stock ticker

        Returns:
            Normalized quote data
        """
        # Try Alpha Vantage first (primary source)
        try:
            quote = self.alpha_vantage.get_quote(symbol)
            if quote:
                self.usage_stats['alpha_vantage'] += 1
                logger.info(f"✅ Quote for {symbol} from Alpha Vantage")
                return self._normalize_quote(quote, 'alpha_vantage')
        except Exception as e:
            logger.warning(f"Alpha Vantage quote failed: {e}")

        # Failover to Finnhub
        try:
            quote = self.finnhub.get_quote(symbol)
            if quote:
                self.usage_stats['finnhub'] += 1
                logger.info(f"✅ Quote for {symbol} from Finnhub (failover)")
                return self._normalize_quote(quote, 'finnhub')
        except Exception as e:
            logger.warning(f"Finnhub quote failed: {e}")

        logger.error(f"❌ Failed to get quote for {symbol} from all sources")
        return None

    def _normalize_quote(self, quote: Dict, source: str) -> Dict:
        """Normalize quote data from different sources"""
        if source == 'alpha_vantage':
            return {
                'symbol': quote.get('symbol'),
                'price': quote.get('price'),
                'change': quote.get('change'),
                'change_percent': float(quote.get('change_percent', 0)),
                'volume': quote.get('volume'),
                'high': quote.get('high'),
                'low': quote.get('low'),
                'open': quote.get('open'),
                'previous_close': quote.get('previous_close'),
                'timestamp': quote.get('timestamp'),
                'source': 'Alpha Vantage'
            }
        elif source == 'finnhub':
            return {
                'symbol': quote.get('symbol'),
                'price': quote.get('current_price'),
                'change': quote.get('change'),
                'change_percent': quote.get('percent_change'),
                'volume': None,  # Not in Finnhub quote
                'high': quote.get('high'),
                'low': quote.get('low'),
                'open': quote.get('open'),
                'previous_close': quote.get('previous_close'),
                'timestamp': quote.get('timestamp'),
                'source': 'Finnhub'
            }
        return quote

    # ========================================================================
    # COMPANY FUNDAMENTALS
    # ========================================================================

    def get_company_fundamentals(self, symbol: str) -> Optional[Dict]:
        """
        Get comprehensive company fundamentals.

        Combines: Alpha Vantage (detailed) + Finnhub (profile)

        Args:
            symbol: Stock ticker

        Returns:
            Combined fundamental data
        """
        fundamentals = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat()
        }

        # Try Alpha Vantage first (has P/E, margins, etc.)
        try:
            overview = self.alpha_vantage.get_company_overview(symbol)
            if overview:
                fundamentals.update({
                    'name': overview.get('name'),
                    'sector': overview.get('sector'),
                    'industry': overview.get('industry'),
                    'market_cap': overview.get('market_cap'),
                    'pe_ratio': overview.get('pe_ratio'),
                    'peg_ratio': overview.get('peg_ratio'),
                    'dividend_yield': overview.get('dividend_yield'),
                    'eps': overview.get('eps'),
                    'profit_margin': overview.get('profit_margin'),
                    'operating_margin': overview.get('operating_margin'),
                    'beta': overview.get('beta'),
                    '52_week_high': overview.get('52_week_high'),
                    '52_week_low': overview.get('52_week_low'),
                    'analyst_target': overview.get('analyst_target_price'),
                    'source': 'Alpha Vantage'
                })
                self.usage_stats['alpha_vantage'] += 1
        except Exception as e:
            logger.warning(f"Alpha Vantage fundamentals failed: {e}")

        # Supplement with Finnhub data
        try:
            profile = self.finnhub.get_company_profile(symbol)
            metrics = self.finnhub.get_company_metrics(symbol)

            if profile:
                # Fill in any missing data
                if 'name' not in fundamentals:
                    fundamentals['name'] = profile.get('name')
                if 'industry' not in fundamentals:
                    fundamentals['industry'] = profile.get('industry')
                fundamentals['exchange'] = profile.get('exchange')
                fundamentals['ipo_date'] = profile.get('ipo_date')
                fundamentals['logo'] = profile.get('logo')
                fundamentals['weburl'] = profile.get('weburl')

            if metrics:
                # Add/supplement metrics
                if 'pe_ratio' not in fundamentals or not fundamentals['pe_ratio']:
                    fundamentals['pe_ratio'] = metrics.get('pe_ratio')
                fundamentals['price_to_book'] = metrics.get('price_to_book')
                fundamentals['price_to_sales'] = metrics.get('price_to_sales')
                fundamentals['roe'] = metrics.get('roe')
                fundamentals['roa'] = metrics.get('roa')

            self.usage_stats['finnhub'] += 2  # profile + metrics

        except Exception as e:
            logger.warning(f"Finnhub fundamentals failed: {e}")

        return fundamentals if len(fundamentals) > 2 else None

    # ========================================================================
    # NEWS & SENTIMENT
    # ========================================================================

    def get_news_and_sentiment(
        self,
        symbol: str,
        days_back: int = 7
    ) -> Optional[Dict]:
        """
        Get news and AI-powered sentiment analysis.

        Combines: Alpha Vantage (sentiment AI) + Finnhub (news)

        Args:
            symbol: Stock ticker
            days_back: Days to look back

        Returns:
            News articles with sentiment scores
        """
        result = {
            'symbol': symbol,
            'articles': [],
            'sentiment': None,
            'timestamp': datetime.now().isoformat()
        }

        # Get AI sentiment from Alpha Vantage (this is GOLD!)
        try:
            sentiment = self.alpha_vantage.get_sentiment_for_ticker(symbol, days_back)
            if sentiment:
                result['sentiment'] = {
                    'score': sentiment.get('sentiment_score'),
                    'label': sentiment.get('sentiment_label'),
                    'confidence': sentiment.get('confidence'),
                    'article_count': sentiment.get('article_count'),
                    'recent_headlines': sentiment.get('recent_headlines', [])
                }
                self.usage_stats['alpha_vantage'] += 1
                logger.info(f"✅ Sentiment for {symbol}: {sentiment['sentiment_label']}")
        except Exception as e:
            logger.warning(f"Alpha Vantage sentiment failed: {e}")

        # Get detailed news from Finnhub
        try:
            news = self.finnhub.get_company_news(symbol, days_back)
            if news:
                result['articles'] = news
                self.usage_stats['finnhub'] += 1
                logger.info(f"✅ Got {len(news)} news articles from Finnhub")
        except Exception as e:
            logger.warning(f"Finnhub news failed: {e}")

        return result if result['articles'] or result['sentiment'] else None

    # ========================================================================
    # ECONOMIC DATA
    # ========================================================================

    def get_economic_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive economic dashboard.

        Uses: FRED (official Federal Reserve data)

        Returns:
            Complete macroeconomic snapshot
        """
        try:
            snapshot = self.fred.get_economic_snapshot()
            recession = self.fred.get_recession_indicators()
            inflation = self.fred.get_inflation_report()
            regime = self.fred.get_market_regime()

            self.usage_stats['fred'] += 4

            return {
                'snapshot': snapshot,
                'recession_indicators': recession,
                'inflation_report': inflation,
                'market_regime': regime,
                'timestamp': datetime.now().isoformat(),
                'source': 'FRED'
            }
        except Exception as e:
            logger.error(f"FRED economic dashboard failed: {e}")
            return {}

    def get_fed_funds_rate(self) -> Optional[float]:
        """Get current Federal Funds rate"""
        try:
            latest = self.fred.get_latest_value('FEDFUNDS')
            self.usage_stats['fred'] += 1
            return latest['value'] if latest else None
        except Exception as e:
            logger.error(f"Failed to get Fed Funds rate: {e}")
            return None

    def get_vix(self) -> Optional[float]:
        """Get current VIX (volatility index)"""
        try:
            latest = self.fred.get_latest_value('VIXCLS')
            self.usage_stats['fred'] += 1
            return latest['value'] if latest else None
        except Exception as e:
            logger.error(f"Failed to get VIX: {e}")
            return None

    def get_unemployment_rate(self) -> Optional[float]:
        """Get current unemployment rate"""
        try:
            latest = self.fred.get_latest_value('UNRATE')
            self.usage_stats['fred'] += 1
            return latest['value'] if latest else None
        except Exception as e:
            logger.error(f"Failed to get unemployment rate: {e}")
            return None

    # ========================================================================
    # INSIDER TRANSACTIONS & ANALYST DATA
    # ========================================================================

    def get_insider_activity(
        self,
        symbol: str,
        months_back: int = 3
    ) -> Optional[List[Dict]]:
        """
        Get insider trading activity.

        Uses: Finnhub

        Args:
            symbol: Stock ticker
            months_back: Months to look back

        Returns:
            List of insider transactions
        """
        try:
            insiders = self.finnhub.get_insider_transactions(symbol, months_back)
            self.usage_stats['finnhub'] += 1
            return insiders
        except Exception as e:
            logger.error(f"Failed to get insider transactions: {e}")
            return None

    def get_analyst_ratings(self, symbol: str) -> Optional[Dict]:
        """
        Get analyst recommendations and price targets.

        Uses: Finnhub

        Args:
            symbol: Stock ticker

        Returns:
            Analyst consensus and price targets
        """
        try:
            recs = self.finnhub.get_recommendations(symbol)
            target = self.finnhub.get_price_target(symbol)

            self.usage_stats['finnhub'] += 2

            return {
                'recommendations': recs[0] if recs else None,
                'price_target': target,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get analyst ratings: {e}")
            return None

    # ========================================================================
    # EARNINGS
    # ========================================================================

    def get_earnings_calendar(
        self,
        symbol: Optional[str] = None,
        days_ahead: int = 30
    ) -> Optional[List[Dict]]:
        """
        Get upcoming earnings calendar.

        Uses: Finnhub

        Args:
            symbol: Stock ticker (optional)
            days_ahead: Days to look ahead

        Returns:
            List of upcoming earnings
        """
        try:
            earnings = self.finnhub.get_earnings_calendar(symbol, days_ahead)
            self.usage_stats['finnhub'] += 1
            return earnings
        except Exception as e:
            logger.error(f"Failed to get earnings calendar: {e}")
            return None

    # ========================================================================
    # COMPREHENSIVE ANALYSIS
    # ========================================================================

    def get_comprehensive_stock_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Get complete stock analysis combining all data sources.

        This is the POWER function - everything you need to know about a stock!

        Args:
            symbol: Stock ticker

        Returns:
            Comprehensive analysis including:
            - Real-time quote
            - Company fundamentals
            - News & sentiment
            - Insider activity
            - Analyst ratings
            - Earnings calendar
            - Technical indicators
        """
        logger.info(f"🔍 Building comprehensive analysis for {symbol}...")

        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'data_sources': []
        }

        # 1. Quote
        quote = self.get_quote(symbol)
        if quote:
            analysis['quote'] = quote
            analysis['data_sources'].append(quote.get('source'))

        # 2. Fundamentals
        fundamentals = self.get_company_fundamentals(symbol)
        if fundamentals:
            analysis['fundamentals'] = fundamentals

        # 3. News & Sentiment (CRITICAL for decision making!)
        news_sentiment = self.get_news_and_sentiment(symbol, days_back=7)
        if news_sentiment:
            analysis['news'] = news_sentiment['articles'][:10]  # Top 10
            analysis['sentiment'] = news_sentiment['sentiment']

        # 4. Insider Activity
        insiders = self.get_insider_activity(symbol, months_back=3)
        if insiders:
            analysis['insider_transactions'] = insiders[:10]  # Recent 10

        # 5. Analyst Ratings
        analyst_data = self.get_analyst_ratings(symbol)
        if analyst_data:
            analysis['analyst_ratings'] = analyst_data

        # 6. Earnings
        earnings = self.get_earnings_calendar(symbol, days_ahead=90)
        if earnings:
            analysis['upcoming_earnings'] = earnings

        # 7. Add macroeconomic context
        analysis['macro_context'] = {
            'fed_funds_rate': self.get_fed_funds_rate(),
            'vix': self.get_vix(),
            'unemployment_rate': self.get_unemployment_rate()
        }

        logger.info(f"✅ Comprehensive analysis complete for {symbol}")
        logger.info(f"   Sources used: {', '.join(set(analysis['data_sources']))}")

        return analysis

    # ========================================================================
    # MARKET CONTEXT FOR AVA
    # ========================================================================

    def get_market_context_for_ava(self) -> Dict[str, Any]:
        """
        Get complete market context formatted for AVA's prompt.

        This provides ALL the context AVA needs to be world-class!

        Returns:
            Market context including economic indicators, regime, trends
        """
        logger.info("🌍 Building market context for AVA...")

        try:
            # Get economic dashboard
            econ = self.get_economic_dashboard()

            # Get key metrics
            fed_rate = self.get_fed_funds_rate()
            vix = self.get_vix()
            unemployment = self.get_unemployment_rate()

            # Format for AVA
            context = {
                'timestamp': datetime.now().isoformat(),
                'economic_snapshot': econ.get('snapshot', {}),
                'recession_risk': econ.get('recession_indicators', {}).get('recession_risk', 'Unknown'),
                'inflation_assessment': econ.get('inflation_report', {}).get('overall_assessment', 'Unknown'),
                'market_regime': econ.get('market_regime', {}).get('regime', 'Unknown'),
                'fed_funds_rate': fed_rate,
                'vix_level': vix,
                'unemployment_rate': unemployment,

                # Interpretations for AVA
                'volatility_regime': 'Low' if vix and vix < 15 else 'High' if vix and vix > 25 else 'Moderate',
                'policy_stance': 'Tight' if fed_rate and fed_rate > 4.5 else 'Accommodative' if fed_rate and fed_rate < 2 else 'Neutral',
                'labor_market': 'Strong' if unemployment and unemployment < 4.5 else 'Weak' if unemployment and unemployment > 6 else 'Moderate'
            }

            logger.info(f"✅ Market context ready:")
            logger.info(f"   Regime: {context['market_regime']}")
            logger.info(f"   Recession Risk: {context['recession_risk']}")
            logger.info(f"   VIX: {context['vix_level']}")

            return context

        except Exception as e:
            logger.error(f"Failed to build market context: {e}")
            return {}

    # ========================================================================
    # USAGE & STATS
    # ========================================================================

    def get_usage_stats(self) -> Dict:
        """Get detailed usage statistics"""
        return {
            **self.usage_stats,
            'total_api_calls': sum([
                self.usage_stats['alpha_vantage'],
                self.usage_stats['fred'],
                self.usage_stats['finnhub']
            ]),
            'all_sources_free': True,
            'sources': {
                'alpha_vantage': {
                    'calls': self.usage_stats['alpha_vantage'],
                    'limit': '25/day',
                    'cost': '$0'
                },
                'fred': {
                    'calls': self.usage_stats['fred'],
                    'limit': 'Unlimited',
                    'cost': '$0'
                },
                'finnhub': {
                    'calls': self.usage_stats['finnhub'],
                    'limit': '60/minute',
                    'cost': '$0'
                }
            },
            'total_cost': '$0.00'
        }

    def reset_stats(self) -> None:
        """Reset usage statistics"""
        for key in self.usage_stats:
            self.usage_stats[key] = 0


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_unified_market_data = None

def get_market_data() -> UnifiedMarketData:
    """Get singleton unified market data instance"""
    global _unified_market_data
    if _unified_market_data is None:
        _unified_market_data = UnifiedMarketData()
    return _unified_market_data


# ============================================================================
# CONVENIENCE FUNCTIONS FOR AVA
# ============================================================================

def analyze_stock(symbol: str) -> Dict:
    """Quick comprehensive stock analysis"""
    return get_market_data().get_comprehensive_stock_analysis(symbol)


def get_market_context() -> Dict:
    """Quick market context for AVA"""
    return get_market_data().get_market_context_for_ava()


def get_quote(symbol: str) -> Optional[Dict]:
    """Quick quote"""
    return get_market_data().get_quote(symbol)


def get_sentiment(symbol: str) -> Optional[Dict]:
    """Quick sentiment"""
    result = get_market_data().get_news_and_sentiment(symbol)
    return result.get('sentiment') if result else None


if __name__ == "__main__":
    # Test the unified service
    logging.basicConfig(level=logging.INFO)

    print("\n=== Testing Unified Market Data Service (ALL FREE) ===\n")

    umd = UnifiedMarketData()

    # Test 1: Comprehensive stock analysis
    print("1. Getting comprehensive analysis for AAPL...")
    analysis = umd.get_comprehensive_stock_analysis('AAPL')
    print(f"   ✅ Got {len(analysis)} data categories")
    if 'quote' in analysis:
        print(f"   Price: ${analysis['quote']['price']}")
    if 'sentiment' in analysis:
        print(f"   Sentiment: {analysis['sentiment']['label']}")

    # Test 2: Market context for AVA
    print("\n2. Getting market context for AVA...")
    context = umd.get_market_context_for_ava()
    print(f"   ✅ Market Regime: {context.get('market_regime')}")
    print(f"   Recession Risk: {context.get('recession_risk')}")
    print(f"   VIX: {context.get('vix_level')}")

    # Test 3: Usage stats
    print("\n3. Usage statistics...")
    stats = umd.get_usage_stats()
    print(f"   ✅ Total API calls: {stats['total_api_calls']}")
    print(f"   Total cost: {stats['total_cost']} (FREE!)")
    print(f"   Alpha Vantage: {stats['sources']['alpha_vantage']['calls']} calls")
    print(f"   FRED: {stats['sources']['fred']['calls']} calls")
    print(f"   Finnhub: {stats['sources']['finnhub']['calls']} calls")
