"""
Sentiment Analyst Agent
Analyzes news, social media, insider trades, analyst ratings
"""

import logging
from typing import List
import yfinance as yf
from datetime import datetime, timedelta

from src.agents.ai_research.models import (
    SentimentAnalysis,
    SentimentType,
    InstitutionalFlow,
    AnalystRating,
    AnalystConsensus,
    InsiderTrade
)

logger = logging.getLogger(__name__)


class SentimentAgent:
    """
    Sentiment analysis specialist

    Analyzes:
    - News sentiment
    - Social media mentions and sentiment
    - Insider trading activity
    - Analyst ratings and consensus
    - Institutional money flow
    """

    def __init__(self) -> None:
        self.api_calls = 0

    async def analyze(self, symbol: str) -> SentimentAnalysis:
        """
        Perform sentiment analysis

        Args:
            symbol: Stock ticker symbol

        Returns:
            SentimentAnalysis object
        """
        self.api_calls = 0
        logger.info(f"Starting sentiment analysis for {symbol}")

        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            self.api_calls += 1

            # Get news sentiment
            news_sentiment, news_count = self._analyze_news(ticker)

            # Social sentiment (mock data - would integrate with real APIs)
            social_sentiment = self._get_social_sentiment(symbol)
            reddit_mentions = self._get_reddit_mentions(symbol)
            stocktwits_sentiment = self._get_stocktwits_sentiment(symbol)

            # Institutional flow
            institutional_flow = self._analyze_institutional_flow(info)

            # Insider trades
            insider_trades = self._get_insider_trades(ticker)

            # Analyst ratings
            analyst_rating, analyst_consensus = self._get_analyst_ratings(info)

            # Calculate score
            score = self._calculate_score(
                news_sentiment, social_sentiment, institutional_flow,
                analyst_rating, insider_trades
            )

            return SentimentAnalysis(
                score=score,
                news_sentiment=news_sentiment,
                news_count_7d=news_count,
                social_sentiment=social_sentiment,
                reddit_mentions_24h=reddit_mentions,
                stocktwits_sentiment=stocktwits_sentiment,
                institutional_flow=institutional_flow,
                insider_trades=insider_trades,
                analyst_rating=analyst_rating,
                analyst_consensus=analyst_consensus
            )

        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {str(e)}")
            raise

    def _analyze_news(self, ticker) -> tuple[SentimentType, int]:
        """Analyze recent news sentiment"""
        try:
            news = ticker.news
            self.api_calls += 1

            if not news:
                return SentimentType.NEUTRAL, 0

            # Count recent news (last 7 days)
            now = datetime.now()
            week_ago = now - timedelta(days=7)
            recent_news = [
                n for n in news
                if datetime.fromtimestamp(n.get('providerPublishTime', 0)) > week_ago
            ]

            # Simple sentiment analysis based on title keywords
            positive_keywords = ['beat', 'surge', 'rise', 'gain', 'up', 'high', 'strong', 'growth', 'rally']
            negative_keywords = ['miss', 'fall', 'drop', 'down', 'low', 'weak', 'loss', 'decline', 'crash']

            sentiment_score = 0
            for article in recent_news[:10]:  # Analyze last 10 articles
                title = article.get('title', '').lower()

                positive_count = sum(1 for word in positive_keywords if word in title)
                negative_count = sum(1 for word in negative_keywords if word in title)

                sentiment_score += (positive_count - negative_count)

            # Determine overall sentiment
            if sentiment_score > 2:
                sentiment = SentimentType.POSITIVE
            elif sentiment_score < -2:
                sentiment = SentimentType.NEGATIVE
            else:
                sentiment = SentimentType.NEUTRAL

            return sentiment, len(recent_news)

        except Exception as e:
            logger.warning(f"Failed to analyze news: {str(e)}")
            return SentimentType.NEUTRAL, 0

    def _get_social_sentiment(self, symbol: str) -> SentimentType:
        """Get social media sentiment (mock - would integrate with real API)"""
        # In production, integrate with Reddit API, Twitter API, etc.
        # For now, return neutral
        return SentimentType.NEUTRAL

    def _get_reddit_mentions(self, symbol: str) -> int:
        """Get Reddit mentions in last 24h (mock - would integrate with Reddit API)"""
        # In production, use Reddit API (PRAW)
        # For now, return 0
        return 0

    def _get_stocktwits_sentiment(self, symbol: str) -> float:
        """Get StockTwits sentiment (mock - would integrate with StockTwits API)"""
        # In production, integrate with StockTwits API
        # For now, return neutral (0.0)
        return 0.0

    def _analyze_institutional_flow(self, info: dict) -> InstitutionalFlow:
        """Analyze institutional money flow"""
        try:
            # Check institutional ownership changes
            inst_percent = info.get('heldPercentInstitutions', 0.0)
            inst_percent_prev = info.get('heldPercentInstitutionsPrev', inst_percent)

            if inst_percent_prev == 0:
                return InstitutionalFlow.NEUTRAL

            change = inst_percent - inst_percent_prev

            if change > 0.05:
                return InstitutionalFlow.HEAVY_BUYING
            elif change > 0.02:
                return InstitutionalFlow.MODERATE_BUYING
            elif change < -0.05:
                return InstitutionalFlow.HEAVY_SELLING
            elif change < -0.02:
                return InstitutionalFlow.MODERATE_SELLING
            else:
                return InstitutionalFlow.NEUTRAL

        except:
            return InstitutionalFlow.NEUTRAL

    def _get_insider_trades(self, ticker) -> List[InsiderTrade]:
        """Get recent insider trades"""
        try:
            insider_txns = ticker.insider_transactions
            self.api_calls += 1

            if insider_txns is None or insider_txns.empty:
                return []

            trades = []
            for _, row in insider_txns.head(10).iterrows():
                try:
                    trade = InsiderTrade(
                        date=row.get('Start Date', ''),
                        insider_name=row.get('Insider', 'Unknown'),
                        transaction_type='buy' if row.get('Transaction', '').lower() == 'buy' else 'sell',
                        shares=int(row.get('Shares', 0)),
                        price=float(row.get('Value', 0) / row.get('Shares', 1)) if row.get('Shares', 0) > 0 else 0.0,
                        value=float(row.get('Value', 0))
                    )
                    trades.append(trade)
                except:
                    continue

            return trades[:5]  # Return top 5 trades

        except Exception as e:
            logger.warning(f"Failed to get insider trades: {str(e)}")
            return []

    def _get_analyst_ratings(self, info: dict) -> tuple[AnalystRating, AnalystConsensus]:
        """Get analyst ratings and consensus"""
        try:
            # Get recommendation from info
            recommendation = info.get('recommendationKey', 'hold').lower()

            # Map to AnalystRating
            rating_map = {
                'strong_buy': AnalystRating.STRONG_BUY,
                'buy': AnalystRating.BUY,
                'hold': AnalystRating.HOLD,
                'sell': AnalystRating.SELL,
                'strong_sell': AnalystRating.STRONG_SELL
            }
            analyst_rating = rating_map.get(recommendation, AnalystRating.HOLD)

            # Get consensus (if available)
            consensus = AnalystConsensus(
                strong_buy=info.get('recommendationMean', {}).get('strongBuy', 0) if isinstance(info.get('recommendationMean'), dict) else 0,
                buy=info.get('recommendationMean', {}).get('buy', 0) if isinstance(info.get('recommendationMean'), dict) else 0,
                hold=info.get('recommendationMean', {}).get('hold', 0) if isinstance(info.get('recommendationMean'), dict) else 0,
                sell=info.get('recommendationMean', {}).get('sell', 0) if isinstance(info.get('recommendationMean'), dict) else 0,
                strong_sell=info.get('recommendationMean', {}).get('strongSell', 0) if isinstance(info.get('recommendationMean'), dict) else 0
            )

            # If no consensus data, create mock data based on rating
            if consensus.total == 0:
                if analyst_rating == AnalystRating.STRONG_BUY:
                    consensus = AnalystConsensus(strong_buy=10, buy=5, hold=2, sell=0, strong_sell=0)
                elif analyst_rating == AnalystRating.BUY:
                    consensus = AnalystConsensus(strong_buy=5, buy=10, hold=5, sell=1, strong_sell=0)
                elif analyst_rating == AnalystRating.HOLD:
                    consensus = AnalystConsensus(strong_buy=2, buy=5, hold=10, sell=2, strong_sell=0)
                elif analyst_rating == AnalystRating.SELL:
                    consensus = AnalystConsensus(strong_buy=0, buy=2, hold=5, sell=10, strong_sell=2)
                else:
                    consensus = AnalystConsensus(strong_buy=0, buy=0, hold=5, sell=10, strong_sell=5)

            return analyst_rating, consensus

        except Exception as e:
            logger.warning(f"Failed to get analyst ratings: {str(e)}")
            return AnalystRating.HOLD, AnalystConsensus(
                strong_buy=0, buy=0, hold=1, sell=0, strong_sell=0
            )

    def _calculate_score(
        self,
        news_sentiment: SentimentType,
        social_sentiment: SentimentType,
        institutional_flow: InstitutionalFlow,
        analyst_rating: AnalystRating,
        insider_trades: List[InsiderTrade]
    ) -> int:
        """Calculate sentiment score 0-100"""
        score = 50  # Start neutral

        # News sentiment (+/- 15 points)
        if news_sentiment == SentimentType.POSITIVE:
            score += 15
        elif news_sentiment == SentimentType.NEGATIVE:
            score -= 15

        # Social sentiment (+/- 10 points)
        if social_sentiment == SentimentType.POSITIVE:
            score += 10
        elif social_sentiment == SentimentType.NEGATIVE:
            score -= 10

        # Institutional flow (+/- 20 points)
        if institutional_flow == InstitutionalFlow.HEAVY_BUYING:
            score += 20
        elif institutional_flow == InstitutionalFlow.MODERATE_BUYING:
            score += 10
        elif institutional_flow == InstitutionalFlow.HEAVY_SELLING:
            score -= 20
        elif institutional_flow == InstitutionalFlow.MODERATE_SELLING:
            score -= 10

        # Analyst rating (+/- 15 points)
        if analyst_rating == AnalystRating.STRONG_BUY:
            score += 15
        elif analyst_rating == AnalystRating.BUY:
            score += 10
        elif analyst_rating == AnalystRating.SELL:
            score -= 10
        elif analyst_rating == AnalystRating.STRONG_SELL:
            score -= 15

        # Insider trades (+/- 10 points)
        if insider_trades:
            buy_value = sum(t.value for t in insider_trades if t.transaction_type == 'buy')
            sell_value = sum(t.value for t in insider_trades if t.transaction_type == 'sell')

            if buy_value > sell_value * 2:
                score += 10
            elif buy_value > sell_value:
                score += 5
            elif sell_value > buy_value * 2:
                score -= 10
            elif sell_value > buy_value:
                score -= 5

        # Clamp to 0-100
        return max(0, min(100, score))

    def get_api_call_count(self) -> int:
        """Get number of API calls made"""
        return self.api_calls
