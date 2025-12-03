"""
AI Flow Analyzer - Intelligent options flow analysis using AI

This module provides AI-powered analysis of options flow data to generate
actionable trading recommendations for the wheel strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import yfinance as yf
import logging
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
import json

# LangChain imports
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class FlowAnalysisResult:
    """Flow analysis result structure"""
    symbol: str
    opportunity_score: float
    best_action: str
    risk_level: str
    confidence: float
    ai_recommendation: str
    key_insights: List[str]
    recommended_strike: Optional[float] = None
    expected_premium: Optional[float] = None
    win_probability: Optional[float] = None


class AIFlowAnalyzer:
    """AI-powered options flow analyzer"""

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize AI Flow Analyzer

        Args:
            openai_api_key: OpenAI API key for GPT-4 integration
        """
        self.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if self.api_key:
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                model="gpt-4-turbo-preview",
                temperature=0.2,
                streaming=False
            )
        else:
            self.llm = None
            logger.warning("No OpenAI API key provided. AI recommendations will be rule-based.")

        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres123!'),
            'database': os.getenv('DB_NAME', 'magnus')
        }

    def get_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def analyze_flow_sentiment(self, flow_data: Dict[str, Any]) -> str:
        """
        Determine bullish/bearish sentiment from flow data

        Args:
            flow_data: Dictionary containing flow metrics

        Returns:
            Sentiment string: 'Bullish', 'Bearish', or 'Neutral'
        """
        try:
            put_call_ratio = flow_data.get('put_call_ratio', 1.0)
            net_flow = flow_data.get('net_premium_flow', 0)
            call_premium = flow_data.get('call_premium', 0)
            put_premium = flow_data.get('put_premium', 0)

            # Weighted scoring system
            sentiment_score = 0

            # Put/Call ratio (most important)
            if put_call_ratio < 0.6:
                sentiment_score += 3  # Very bullish
            elif put_call_ratio < 0.8:
                sentiment_score += 2  # Moderately bullish
            elif put_call_ratio < 1.2:
                sentiment_score += 0  # Neutral
            elif put_call_ratio < 1.5:
                sentiment_score -= 2  # Moderately bearish
            else:
                sentiment_score -= 3  # Very bearish

            # Net premium flow
            total_premium = call_premium + put_premium
            if total_premium > 0:
                net_flow_pct = net_flow / total_premium
                if net_flow_pct > 0.3:
                    sentiment_score += 2
                elif net_flow_pct > 0.1:
                    sentiment_score += 1
                elif net_flow_pct < -0.3:
                    sentiment_score -= 2
                elif net_flow_pct < -0.1:
                    sentiment_score -= 1

            # Determine final sentiment
            if sentiment_score >= 3:
                return 'Bullish'
            elif sentiment_score <= -3:
                return 'Bearish'
            else:
                return 'Neutral'

        except Exception as e:
            logger.error(f"Error analyzing flow sentiment: {e}")
            return 'Neutral'

    def generate_flow_insights(self, symbol: str, flow_data: Dict[str, Any],
                              historical_data: List[Dict[str, Any]]) -> FlowAnalysisResult:
        """
        Generate comprehensive AI insights for options flow

        Args:
            symbol: Stock ticker symbol
            flow_data: Current flow data
            historical_data: Historical flow data

        Returns:
            FlowAnalysisResult object
        """
        try:
            # Get additional market data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))

            # Calculate opportunity score
            opportunity_score = self.score_flow_opportunity(symbol, flow_data)

            # Recommend best action
            best_action = self.recommend_best_action(flow_data, historical_data)

            # Assess risk
            risk_level = self.assess_flow_risk(symbol, flow_data)

            # Generate insights
            if self.llm:
                ai_recommendation, key_insights, confidence = self._generate_llm_insights(
                    symbol, flow_data, historical_data, info
                )
            else:
                ai_recommendation, key_insights, confidence = self._generate_rule_based_insights(
                    symbol, flow_data, historical_data, info
                )

            # Calculate recommended strike and premium
            recommended_strike, expected_premium, win_prob = self._calculate_recommendations(
                symbol, current_price, flow_data, best_action
            )

            return FlowAnalysisResult(
                symbol=symbol,
                opportunity_score=opportunity_score,
                best_action=best_action,
                risk_level=risk_level,
                confidence=confidence,
                ai_recommendation=ai_recommendation,
                key_insights=key_insights,
                recommended_strike=recommended_strike,
                expected_premium=expected_premium,
                win_probability=win_prob
            )

        except Exception as e:
            logger.error(f"Error generating flow insights for {symbol}: {e}")
            return FlowAnalysisResult(
                symbol=symbol,
                opportunity_score=0,
                best_action='WAIT',
                risk_level='High',
                confidence=0,
                ai_recommendation='Error generating insights',
                key_insights=['Analysis failed']
            )

    def score_flow_opportunity(self, symbol: str, flow_metrics: Dict[str, Any]) -> float:
        """
        Calculate 0-100 opportunity score

        Args:
            symbol: Stock ticker symbol
            flow_metrics: Flow metrics dictionary

        Returns:
            Opportunity score (0-100)
        """
        try:
            score = 0.0

            # 1. Volume analysis (0-25 points)
            volume_percentile = flow_metrics.get('volume_percentile', 50)
            score += min(volume_percentile / 4, 25)

            # 2. Flow sentiment consistency (0-25 points)
            sentiment = flow_metrics.get('flow_sentiment', 'Neutral')
            put_call_ratio = flow_metrics.get('put_call_ratio', 1.0)

            if sentiment == 'Bullish' and put_call_ratio < 0.7:
                score += 25
            elif sentiment == 'Bearish' and put_call_ratio > 1.3:
                score += 25
            elif sentiment == 'Neutral':
                score += 10

            # 3. Trend strength (0-25 points)
            trend = flow_metrics.get('trend_7d', 'Stable')
            net_flow_7d = flow_metrics.get('net_flow_7d', 0)

            if trend == 'Increasing' and net_flow_7d > 0:
                score += 25
            elif trend == 'Decreasing' and net_flow_7d < 0:
                score += 25
            elif trend == 'Stable':
                score += 15

            # 4. Unusual activity bonus (0-25 points)
            if flow_metrics.get('unusual_activity', False):
                score += 20

            # Consistency check - reduce score if metrics conflict
            consistency_penalty = 0
            if sentiment == 'Bullish' and net_flow_7d < 0:
                consistency_penalty += 15
            elif sentiment == 'Bearish' and net_flow_7d > 0:
                consistency_penalty += 15

            score = max(0, score - consistency_penalty)

            return min(score, 100)

        except Exception as e:
            logger.error(f"Error scoring opportunity for {symbol}: {e}")
            return 0

    def recommend_best_action(self, flow_data: Dict[str, Any],
                             historical_data: List[Dict[str, Any]]) -> str:
        """
        Recommend specific action based on flow analysis

        Args:
            flow_data: Current flow data
            historical_data: Historical flow data

        Returns:
            Action string: 'BUY_CALL', 'SELL_PUT', 'BUY_PUT', 'SELL_CALL', 'WAIT'
        """
        try:
            sentiment = flow_data.get('flow_sentiment', 'Neutral')
            put_call_ratio = flow_data.get('put_call_ratio', 1.0)
            net_flow = flow_data.get('net_premium_flow', 0)
            trend = flow_data.get('trend_7d', 'Stable')

            # Decision tree for wheel strategy alignment
            if sentiment == 'Bullish' and put_call_ratio < 0.7:
                # Strong bullish flow - sell cash-secured puts
                if trend == 'Increasing' or trend == 'Stable':
                    return 'SELL_PUT'
                else:
                    return 'WAIT'

            elif sentiment == 'Bearish' and put_call_ratio > 1.3:
                # Strong bearish flow - avoid or buy protective puts
                if net_flow < -50000:
                    return 'BUY_PUT'
                else:
                    return 'WAIT'

            elif sentiment == 'Neutral':
                # Neutral flow - look for theta decay opportunities
                if put_call_ratio > 0.9 and put_call_ratio < 1.1:
                    return 'SELL_PUT'
                else:
                    return 'WAIT'

            else:
                return 'WAIT'

        except Exception as e:
            logger.error(f"Error recommending action: {e}")
            return 'WAIT'

    def assess_flow_risk(self, symbol: str, flow_volatility: Dict[str, Any]) -> str:
        """
        Calculate risk level based on flow volatility

        Args:
            symbol: Stock ticker symbol
            flow_volatility: Flow volatility metrics

        Returns:
            Risk level: 'Low', 'Medium', 'High'
        """
        try:
            # Get historical volatility
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='30d')

            if hist.empty:
                return 'High'

            # Calculate price volatility
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized

            # Get flow consistency
            consistency = flow_volatility.get('consistency_score', 50)
            unusual_count = flow_volatility.get('unusual_activity_count_7d', 0)

            # Risk scoring
            risk_score = 0

            # Price volatility
            if volatility > 0.6:
                risk_score += 3
            elif volatility > 0.4:
                risk_score += 2
            else:
                risk_score += 1

            # Flow consistency
            if consistency < 40:
                risk_score += 2
            elif consistency < 60:
                risk_score += 1

            # Unusual activity
            if unusual_count > 3:
                risk_score += 2
            elif unusual_count > 1:
                risk_score += 1

            # Determine risk level
            if risk_score <= 3:
                return 'Low'
            elif risk_score <= 5:
                return 'Medium'
            else:
                return 'High'

        except Exception as e:
            logger.error(f"Error assessing risk for {symbol}: {e}")
            return 'High'

    def _generate_llm_insights(self, symbol: str, flow_data: Dict[str, Any],
                              historical_data: List[Dict[str, Any]],
                              stock_info: Dict[str, Any]) -> Tuple[str, List[str], float]:
        """Generate insights using LLM"""

        system_prompt = """You are an expert options flow analyst specializing in
        identifying institutional trading patterns and generating actionable insights
        for the wheel strategy. Analyze options flow data and provide clear, specific
        recommendations focused on cash-secured puts and covered calls."""

        human_prompt = f"""
        Analyze the options flow for {symbol} and provide insights:

        Current Flow Data:
        - Put/Call Ratio: {flow_data.get('put_call_ratio', 'N/A'):.2f}
        - Net Premium Flow: ${flow_data.get('net_premium_flow', 0):,.0f}
        - Call Premium: ${flow_data.get('call_premium', 0):,.0f}
        - Put Premium: ${flow_data.get('put_premium', 0):,.0f}
        - Total Volume: {flow_data.get('total_volume', 0):,}
        - Flow Sentiment: {flow_data.get('flow_sentiment', 'Unknown')}
        - 7-Day Trend: {flow_data.get('trend_7d', 'Unknown')}

        Stock Information:
        - Current Price: ${stock_info.get('currentPrice', stock_info.get('regularMarketPrice', 'N/A'))}
        - Market Cap: ${stock_info.get('marketCap', 0):,.0f}
        - Sector: {stock_info.get('sector', 'Unknown')}
        - PE Ratio: {stock_info.get('trailingPE', 'N/A')}

        Provide:
        1. One-paragraph trading recommendation for wheel strategy traders
        2. Three key insights as bullet points
        3. Confidence score (0-1) for this analysis

        Format your response as JSON:
        {{
            "recommendation": "Your detailed recommendation here",
            "insights": ["Insight 1", "Insight 2", "Insight 3"],
            "confidence": 0.85
        }}
        """

        try:
            messages = [
                SystemMessagePromptTemplate.from_template(system_prompt).format(),
                HumanMessagePromptTemplate.from_template(human_prompt).format()
            ]

            response = self.llm.predict_messages(messages)
            result = json.loads(response.content)

            return (
                result.get('recommendation', 'No recommendation available'),
                result.get('insights', ['No insights available']),
                float(result.get('confidence', 0.5))
            )

        except Exception as e:
            logger.error(f"Error generating LLM insights: {e}")
            return self._generate_rule_based_insights(symbol, flow_data, historical_data, stock_info)

    def _generate_rule_based_insights(self, symbol: str, flow_data: Dict[str, Any],
                                     historical_data: List[Dict[str, Any]],
                                     stock_info: Dict[str, Any]) -> Tuple[str, List[str], float]:
        """Generate insights using rule-based logic"""

        put_call_ratio = flow_data.get('put_call_ratio', 1.0)
        net_flow = flow_data.get('net_premium_flow', 0)
        sentiment = flow_data.get('flow_sentiment', 'Neutral')
        trend = flow_data.get('trend_7d', 'Unknown')

        insights = []
        confidence = 0.7

        # Generate recommendation
        if sentiment == 'Bullish' and put_call_ratio < 0.7:
            recommendation = f"{symbol} shows strong bullish flow with heavy call buying. "
            recommendation += f"Put/call ratio of {put_call_ratio:.2f} indicates institutional confidence. "
            recommendation += "Consider selling cash-secured puts 5-10% below current price for premium collection."
            insights.append(f"Bullish flow with P/C ratio {put_call_ratio:.2f} - well below neutral")
            insights.append(f"Net premium flow ${net_flow:,.0f} favors calls")
            confidence = 0.8

        elif sentiment == 'Bearish' and put_call_ratio > 1.3:
            recommendation = f"{symbol} exhibits bearish flow with elevated put buying. "
            recommendation += f"Put/call ratio of {put_call_ratio:.2f} suggests caution. "
            recommendation += "Wait for stabilization before entering new positions or consider protective puts."
            insights.append(f"Bearish flow with P/C ratio {put_call_ratio:.2f} - significantly elevated")
            insights.append(f"Heavy put buying of ${abs(net_flow):,.0f}")
            confidence = 0.75

        else:
            recommendation = f"{symbol} shows balanced options flow. "
            recommendation += f"Put/call ratio near {put_call_ratio:.2f} indicates neutral sentiment. "
            recommendation += "Suitable for standard wheel strategy with delta ~0.30 strikes."
            insights.append("Balanced flow suggests neutral market view")
            insights.append(f"P/C ratio {put_call_ratio:.2f} near equilibrium")
            confidence = 0.65

        # Add trend insight
        if trend == 'Increasing':
            insights.append("7-day flow trend is increasing - momentum building")
        elif trend == 'Decreasing':
            insights.append("7-day flow trend declining - momentum fading")
        else:
            insights.append("7-day flow trend stable - consistent pattern")

        # Ensure we have exactly 3 insights
        while len(insights) < 3:
            insights.append(f"Total volume: {flow_data.get('total_volume', 0):,} contracts")

        return recommendation, insights[:3], confidence

    def _calculate_recommendations(self, symbol: str, current_price: float,
                                  flow_data: Dict[str, Any], best_action: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Calculate recommended strike, premium, and win probability"""

        try:
            if best_action == 'WAIT' or current_price == 0:
                return None, None, None

            # Get options data
            ticker = yf.Ticker(symbol)
            expirations = ticker.options

            if not expirations:
                return None, None, None

            # Target 30-45 DTE
            target_exp = None
            for exp in expirations:
                exp_date = datetime.strptime(exp, '%Y-%m-%d')
                dte = (exp_date - datetime.now()).days
                if 30 <= dte <= 45:
                    target_exp = exp
                    break

            if not target_exp:
                target_exp = expirations[0] if expirations else None

            if not target_exp:
                return None, None, None

            # Get options chain
            opt_chain = ticker.option_chain(target_exp)

            if best_action == 'SELL_PUT':
                # Find put with delta ~0.30 (roughly 30% OTM)
                target_strike = current_price * 0.70
                puts = opt_chain.puts

                # Find closest strike
                closest_put = puts.iloc[(puts['strike'] - target_strike).abs().argsort()[:1]]

                if not closest_put.empty:
                    strike = float(closest_put['strike'].values[0])
                    premium = float(closest_put['lastPrice'].values[0]) * 100

                    # Estimate win probability (simplified)
                    win_prob = 0.70 + (current_price - strike) / current_price * 0.2

                    return strike, premium, win_prob

            return None, None, None

        except Exception as e:
            logger.error(f"Error calculating recommendations for {symbol}: {e}")
            return None, None, None

    def save_analysis(self, analysis: FlowAnalysisResult) -> bool:
        """
        Save analysis results to database

        Args:
            analysis: FlowAnalysisResult object

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            # Get current flow data for additional fields
            cur.execute("""
                SELECT put_call_ratio, net_premium_flow
                FROM options_flow
                WHERE symbol = %s AND flow_date = CURRENT_DATE
            """, (analysis.symbol,))

            flow_row = cur.fetchone()

            if flow_row:
                put_call_ratio_7d = flow_row[0]
                net_flow_7d = flow_row[1]
            else:
                put_call_ratio_7d = None
                net_flow_7d = 0

            cur.execute("""
                INSERT INTO options_flow_analysis (
                    symbol, opportunity_score, best_action, risk_level,
                    confidence, ai_recommendation, key_insights,
                    recommended_strike, expected_premium, win_probability,
                    avg_put_call_ratio_7d, net_flow_7d
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (symbol)
                DO UPDATE SET
                    opportunity_score = EXCLUDED.opportunity_score,
                    best_action = EXCLUDED.best_action,
                    risk_level = EXCLUDED.risk_level,
                    confidence = EXCLUDED.confidence,
                    ai_recommendation = EXCLUDED.ai_recommendation,
                    key_insights = EXCLUDED.key_insights,
                    recommended_strike = EXCLUDED.recommended_strike,
                    expected_premium = EXCLUDED.expected_premium,
                    win_probability = EXCLUDED.win_probability,
                    avg_put_call_ratio_7d = EXCLUDED.avg_put_call_ratio_7d,
                    net_flow_7d = EXCLUDED.net_flow_7d,
                    last_updated = NOW()
            """, (
                analysis.symbol, analysis.opportunity_score, analysis.best_action,
                analysis.risk_level, analysis.confidence, analysis.ai_recommendation,
                analysis.key_insights, analysis.recommended_strike,
                analysis.expected_premium, analysis.win_probability,
                put_call_ratio_7d, net_flow_7d
            ))

            conn.commit()
            cur.close()
            conn.close()

            logger.info(f"Saved analysis for {analysis.symbol}")
            return True

        except Exception as e:
            logger.error(f"Error saving analysis for {analysis.symbol}: {e}")
            return False

    def batch_analyze(self, symbols: List[str]) -> int:
        """
        Analyze flow for multiple symbols

        Args:
            symbols: List of ticker symbols

        Returns:
            Number of successful analyses
        """
        success_count = 0

        for symbol in symbols:
            try:
                # Get flow data
                conn = self.get_connection()
                cur = conn.cursor(cursor_factory=RealDictCursor)

                cur.execute("""
                    SELECT *
                    FROM options_flow
                    WHERE symbol = %s
                    ORDER BY flow_date DESC
                    LIMIT 30
                """, (symbol,))

                historical_data = cur.fetchall()
                cur.close()
                conn.close()

                if not historical_data:
                    continue

                current_flow = dict(historical_data[0])

                # Generate analysis
                analysis = self.generate_flow_insights(symbol, current_flow, historical_data)

                # Save to database
                if self.save_analysis(analysis):
                    success_count += 1

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue

        return success_count


if __name__ == "__main__":
    # Test the analyzer
    logging.basicConfig(level=logging.INFO)

    analyzer = AIFlowAnalyzer()

    # Test single symbol
    test_symbol = 'AAPL'
    flow_data = {
        'symbol': test_symbol,
        'put_call_ratio': 0.65,
        'net_premium_flow': 15000000,
        'call_premium': 25000000,
        'put_premium': 10000000,
        'total_volume': 50000,
        'flow_sentiment': 'Bullish',
        'trend_7d': 'Increasing',
        'unusual_activity': True,
        'volume_percentile': 85
    }

    result = analyzer.generate_flow_insights(test_symbol, flow_data, [flow_data])

    print(f"\nFlow Analysis for {test_symbol}:")
    print(f"  Opportunity Score: {result.opportunity_score:.1f}/100")
    print(f"  Best Action: {result.best_action}")
    print(f"  Risk Level: {result.risk_level}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"\n  AI Recommendation:")
    print(f"  {result.ai_recommendation}")
    print(f"\n  Key Insights:")
    for insight in result.key_insights:
        print(f"    - {insight}")
