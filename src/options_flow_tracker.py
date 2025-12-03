"""
Options Flow Tracker - Track institutional options flow and premium inflows/outflows

This module collects and analyzes options volume and premium data from Yahoo Finance
to identify institutional trading patterns and opportunities for the wheel strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, date
import yfinance as yf
import logging
import time
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class OptionsFlowData:
    """Options flow data structure"""
    symbol: str
    flow_date: date
    call_volume: int
    put_volume: int
    call_premium: float
    put_premium: float
    net_premium_flow: float
    put_call_ratio: float
    unusual_activity: bool
    flow_sentiment: str
    avg_call_premium: float
    avg_put_premium: float
    total_volume: int
    total_open_interest: int
    iv_rank: Optional[float]


class OptionsFlowTracker:
    """Track and analyze options flow data"""

    def __init__(self) -> None:
        """Initialize Options Flow Tracker"""
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres123!'),
            'database': os.getenv('DB_NAME', 'magnus')
        }
        self.rate_limit_delay = 0.5  # 500ms between API calls

    def get_connection(self) -> None:
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def fetch_options_flow(self, symbol: str) -> Optional[OptionsFlowData]:
        """
        Fetch options volume and premium data from Yahoo Finance

        Args:
            symbol: Stock ticker symbol

        Returns:
            OptionsFlowData object or None if fetch fails
        """
        try:
            logger.info(f"Fetching options flow for {symbol}")
            ticker = yf.Ticker(symbol)

            # Get all available expiration dates
            expirations = ticker.options
            if not expirations:
                logger.warning(f"No options available for {symbol}")
                return None

            # Initialize accumulators
            total_call_volume = 0
            total_put_volume = 0
            total_call_premium = 0.0
            total_put_premium = 0.0
            total_call_oi = 0
            total_put_oi = 0
            call_count = 0
            put_count = 0

            # Process first 3 expiration dates (near-term flow)
            for exp_date in expirations[:3]:
                try:
                    opt_chain = ticker.option_chain(exp_date)

                    # Process calls
                    if opt_chain.calls is not None and len(opt_chain.calls) > 0:
                        calls = opt_chain.calls
                        calls = calls[calls['volume'].notna() & (calls['volume'] > 0)]

                        for _, row in calls.iterrows():
                            volume = int(row.get('volume', 0))
                            last_price = float(row.get('lastPrice', 0))
                            premium = volume * last_price * 100  # Contract multiplier

                            total_call_volume += volume
                            total_call_premium += premium
                            total_call_oi += int(row.get('openInterest', 0))
                            call_count += 1

                    # Process puts
                    if opt_chain.puts is not None and len(opt_chain.puts) > 0:
                        puts = opt_chain.puts
                        puts = puts[puts['volume'].notna() & (puts['volume'] > 0)]

                        for _, row in puts.iterrows():
                            volume = int(row.get('volume', 0))
                            last_price = float(row.get('lastPrice', 0))
                            premium = volume * last_price * 100  # Contract multiplier

                            total_put_volume += volume
                            total_put_premium += premium
                            total_put_oi += int(row.get('openInterest', 0))
                            put_count += 1

                    time.sleep(0.2)  # Rate limiting between expirations

                except Exception as e:
                    logger.warning(f"Error processing expiration {exp_date} for {symbol}: {e}")
                    continue

            # Calculate metrics
            total_volume = total_call_volume + total_put_volume
            if total_volume == 0:
                logger.warning(f"No volume data for {symbol}")
                return None

            net_premium_flow = total_call_premium - total_put_premium
            put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
            avg_call_premium = total_call_premium / call_count if call_count > 0 else 0
            avg_put_premium = total_put_premium / put_count if put_count > 0 else 0

            # Determine flow sentiment
            if put_call_ratio < 0.7:
                sentiment = 'Bullish'
            elif put_call_ratio > 1.3:
                sentiment = 'Bearish'
            else:
                sentiment = 'Neutral'

            # Check for unusual activity (will be validated against historical average)
            unusual = total_volume > 10000  # Preliminary flag, refined in calculate_flow_metrics

            flow_data = OptionsFlowData(
                symbol=symbol,
                flow_date=date.today(),
                call_volume=total_call_volume,
                put_volume=total_put_volume,
                call_premium=total_call_premium,
                put_premium=total_put_premium,
                net_premium_flow=net_premium_flow,
                put_call_ratio=put_call_ratio,
                unusual_activity=unusual,
                flow_sentiment=sentiment,
                avg_call_premium=avg_call_premium,
                avg_put_premium=avg_put_premium,
                total_volume=total_volume,
                total_open_interest=total_call_oi + total_put_oi,
                iv_rank=None  # Calculate separately if needed
            )

            logger.info(f"Successfully fetched flow for {symbol}: {total_volume} total volume")
            return flow_data

        except Exception as e:
            logger.error(f"Error fetching options flow for {symbol}: {e}")
            return None

    def calculate_flow_metrics(self, symbol: str) -> Dict[str, Any]:
        """
        Calculate flow metrics including trends and unusual activity detection

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary of calculated metrics
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Get historical flow data (last 30 days)
            cur.execute("""
                SELECT
                    flow_date,
                    call_volume,
                    put_volume,
                    total_volume,
                    net_premium_flow,
                    put_call_ratio
                FROM options_flow
                WHERE symbol = %s
                    AND flow_date >= CURRENT_DATE - INTERVAL '30 days'
                ORDER BY flow_date DESC
            """, (symbol,))

            historical_data = cur.fetchall()
            cur.close()
            conn.close()

            if not historical_data or len(historical_data) < 2:
                return {
                    'avg_volume': 0,
                    'volume_percentile': 0,
                    'is_unusual': False,
                    'trend_7d': 'Unknown',
                    'flow_acceleration': 0
                }

            # Calculate average volume
            volumes = [row['total_volume'] for row in historical_data]
            avg_volume = np.mean(volumes)
            std_volume = np.std(volumes)
            current_volume = volumes[0] if volumes else 0

            # Check if current volume is unusual (>2x average or >2 std dev)
            is_unusual = (current_volume > avg_volume * 2) or \
                        (current_volume > avg_volume + 2 * std_volume)

            # Calculate volume percentile
            volume_percentile = (sum(1 for v in volumes if v <= current_volume) / len(volumes)) * 100

            # Calculate 7-day trend
            if len(historical_data) >= 7:
                recent_flows = [row['net_premium_flow'] for row in historical_data[:7]]
                trend_slope = np.polyfit(range(7), recent_flows, 1)[0]

                if trend_slope > avg_volume * 0.1:
                    trend_7d = 'Increasing'
                elif trend_slope < -avg_volume * 0.1:
                    trend_7d = 'Decreasing'
                else:
                    trend_7d = 'Stable'

                flow_acceleration = trend_slope
            else:
                trend_7d = 'Insufficient Data'
                flow_acceleration = 0

            return {
                'avg_volume': avg_volume,
                'volume_percentile': volume_percentile,
                'is_unusual': is_unusual,
                'trend_7d': trend_7d,
                'flow_acceleration': flow_acceleration
            }

        except Exception as e:
            logger.error(f"Error calculating flow metrics for {symbol}: {e}")
            return {
                'avg_volume': 0,
                'volume_percentile': 0,
                'is_unusual': False,
                'trend_7d': 'Error',
                'flow_acceleration': 0
            }

    def analyze_flow_trend(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Analyze flow trend over specified number of days

        Args:
            symbol: Stock ticker symbol
            days: Number of days to analyze (default: 7)

        Returns:
            Dictionary with trend analysis
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT
                    flow_date,
                    net_premium_flow,
                    put_call_ratio,
                    flow_sentiment,
                    total_volume
                FROM options_flow
                WHERE symbol = %s
                    AND flow_date >= CURRENT_DATE - INTERVAL '%s days'
                ORDER BY flow_date ASC
            """, (symbol, days))

            trend_data = cur.fetchall()
            cur.close()
            conn.close()

            if not trend_data or len(trend_data) < 2:
                return {
                    'trend_direction': 'Unknown',
                    'net_flow': 0,
                    'avg_put_call_ratio': 0,
                    'dominant_sentiment': 'Neutral',
                    'consistency_score': 0
                }

            # Calculate net flow
            net_flow = sum(row['net_premium_flow'] for row in trend_data)

            # Calculate average put/call ratio
            avg_pc_ratio = np.mean([row['put_call_ratio'] for row in trend_data])

            # Determine dominant sentiment
            sentiments = [row['flow_sentiment'] for row in trend_data]
            sentiment_counts = pd.Series(sentiments).value_counts()
            dominant_sentiment = sentiment_counts.index[0] if len(sentiment_counts) > 0 else 'Neutral'

            # Calculate trend direction
            flows = [row['net_premium_flow'] for row in trend_data]
            if len(flows) >= 2:
                trend_slope = np.polyfit(range(len(flows)), flows, 1)[0]
                if trend_slope > 0:
                    trend_direction = 'Increasing'
                elif trend_slope < 0:
                    trend_direction = 'Decreasing'
                else:
                    trend_direction = 'Stable'
            else:
                trend_direction = 'Unknown'

            # Calculate consistency (how often sentiment agrees with net flow)
            consistent_days = sum(
                1 for row in trend_data
                if (row['net_premium_flow'] > 0 and row['flow_sentiment'] == 'Bullish') or
                   (row['net_premium_flow'] < 0 and row['flow_sentiment'] == 'Bearish')
            )
            consistency_score = (consistent_days / len(trend_data)) * 100

            return {
                'trend_direction': trend_direction,
                'net_flow': net_flow,
                'avg_put_call_ratio': avg_pc_ratio,
                'dominant_sentiment': dominant_sentiment,
                'consistency_score': consistency_score
            }

        except Exception as e:
            logger.error(f"Error analyzing flow trend for {symbol}: {e}")
            return {
                'trend_direction': 'Error',
                'net_flow': 0,
                'avg_put_call_ratio': 0,
                'dominant_sentiment': 'Neutral',
                'consistency_score': 0
            }

    def detect_unusual_activity(self, symbol: str) -> Tuple[bool, str]:
        """
        Detect unusual options activity

        Args:
            symbol: Stock ticker symbol

        Returns:
            Tuple of (is_unusual, description)
        """
        try:
            metrics = self.calculate_flow_metrics(symbol)

            if metrics['is_unusual']:
                volume_pct = metrics['volume_percentile']
                return True, f"Unusual volume: {volume_pct:.0f}th percentile (2x+ average)"

            return False, "Normal activity levels"

        except Exception as e:
            logger.error(f"Error detecting unusual activity for {symbol}: {e}")
            return False, "Error detecting activity"

    def save_flow_data(self, flow_data: OptionsFlowData) -> bool:
        """
        Save flow data to database

        Args:
            flow_data: OptionsFlowData object

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO options_flow (
                    symbol, flow_date, call_volume, put_volume,
                    call_premium, put_premium, net_premium_flow,
                    put_call_ratio, unusual_activity, flow_sentiment,
                    avg_call_premium, avg_put_premium, total_volume,
                    total_open_interest, iv_rank
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (symbol, flow_date)
                DO UPDATE SET
                    call_volume = EXCLUDED.call_volume,
                    put_volume = EXCLUDED.put_volume,
                    call_premium = EXCLUDED.call_premium,
                    put_premium = EXCLUDED.put_premium,
                    net_premium_flow = EXCLUDED.net_premium_flow,
                    put_call_ratio = EXCLUDED.put_call_ratio,
                    unusual_activity = EXCLUDED.unusual_activity,
                    flow_sentiment = EXCLUDED.flow_sentiment,
                    avg_call_premium = EXCLUDED.avg_call_premium,
                    avg_put_premium = EXCLUDED.avg_put_premium,
                    total_volume = EXCLUDED.total_volume,
                    total_open_interest = EXCLUDED.total_open_interest,
                    iv_rank = EXCLUDED.iv_rank,
                    last_updated = NOW()
            """, (
                flow_data.symbol, flow_data.flow_date, flow_data.call_volume,
                flow_data.put_volume, flow_data.call_premium, flow_data.put_premium,
                flow_data.net_premium_flow, flow_data.put_call_ratio,
                flow_data.unusual_activity, flow_data.flow_sentiment,
                flow_data.avg_call_premium, flow_data.avg_put_premium,
                flow_data.total_volume, flow_data.total_open_interest,
                flow_data.iv_rank
            ))

            conn.commit()
            cur.close()
            conn.close()

            logger.info(f"Saved flow data for {flow_data.symbol}")
            return True

        except Exception as e:
            logger.error(f"Error saving flow data for {flow_data.symbol}: {e}")
            return False

    def batch_update_flow(self, symbols: List[str], limit: int = 100) -> Dict[str, int]:
        """
        Update flow data for multiple symbols

        Args:
            symbols: List of stock ticker symbols
            limit: Maximum number of symbols to process

        Returns:
            Dictionary with success/failure counts
        """
        results = {
            'success': 0,
            'failed': 0,
            'skipped': 0
        }

        symbols_to_process = symbols[:limit]

        for i, symbol in enumerate(symbols_to_process):
            try:
                logger.info(f"Processing {i+1}/{len(symbols_to_process)}: {symbol}")

                # Fetch flow data
                flow_data = self.fetch_options_flow(symbol)

                if flow_data:
                    # Calculate metrics to update unusual activity flag
                    metrics = self.calculate_flow_metrics(symbol)
                    flow_data.unusual_activity = metrics['is_unusual']

                    # Save to database
                    if self.save_flow_data(flow_data):
                        results['success'] += 1
                    else:
                        results['failed'] += 1
                else:
                    results['skipped'] += 1

                # Rate limiting
                time.sleep(self.rate_limit_delay)

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results['failed'] += 1
                continue

        logger.info(f"Batch update complete: {results}")
        return results

    def get_top_flow_opportunities(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get top flow opportunities sorted by score

        Args:
            limit: Maximum number of opportunities to return

        Returns:
            List of opportunity dictionaries
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT
                    ofa.symbol,
                    ofa.opportunity_score,
                    ofa.net_flow_7d,
                    ofa.flow_trend_7d,
                    ofa.dominant_strategy,
                    ofa.best_action,
                    ofa.risk_level,
                    ofa.confidence,
                    ofa.ai_recommendation,
                    ofa.current_price,
                    ofa.recommended_strike,
                    ofa.expected_premium,
                    of.put_call_ratio,
                    of.flow_sentiment,
                    of.total_volume
                FROM options_flow_analysis ofa
                JOIN options_flow of ON ofa.symbol = of.symbol
                WHERE of.flow_date = CURRENT_DATE
                    AND ofa.opportunity_score > 0
                ORDER BY ofa.opportunity_score DESC
                LIMIT %s
            """, (limit,))

            opportunities = cur.fetchall()
            cur.close()
            conn.close()

            return [dict(opp) for opp in opportunities]

        except Exception as e:
            logger.error(f"Error fetching top opportunities: {e}")
            return []

    def get_market_flow_summary(self) -> Dict[str, Any]:
        """
        Get market-wide flow summary for today

        Returns:
            Dictionary with market flow statistics
        """
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT
                    COUNT(*) as total_symbols,
                    SUM(call_premium) as total_call_premium,
                    SUM(put_premium) as total_put_premium,
                    SUM(net_premium_flow) as total_net_flow,
                    AVG(put_call_ratio) as avg_put_call_ratio,
                    COUNT(CASE WHEN flow_sentiment = 'Bullish' THEN 1 END) as bullish_count,
                    COUNT(CASE WHEN flow_sentiment = 'Bearish' THEN 1 END) as bearish_count,
                    COUNT(CASE WHEN flow_sentiment = 'Neutral' THEN 1 END) as neutral_count,
                    COUNT(CASE WHEN unusual_activity = true THEN 1 END) as unusual_count
                FROM options_flow
                WHERE flow_date = CURRENT_DATE
            """)

            summary = cur.fetchone()
            cur.close()
            conn.close()

            return dict(summary) if summary else {}

        except Exception as e:
            logger.error(f"Error fetching market flow summary: {e}")
            return {}


# Standalone utility functions
def get_popular_optionable_symbols() -> List[str]:
    """Get list of popular optionable stocks for tracking"""
    # Common liquid stocks with active options markets
    return [
        # Mega Cap Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD',
        # Finance
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C',
        # Consumer
        'DIS', 'NFLX', 'COST', 'WMT', 'HD', 'MCD', 'NKE',
        # Healthcare
        'UNH', 'JNJ', 'PFE', 'ABBV', 'LLY', 'MRK',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB',
        # Industrial
        'BA', 'CAT', 'GE', 'UPS',
        # Communication
        'T', 'VZ', 'CMCSA',
        # ETFs
        'SPY', 'QQQ', 'IWM', 'DIA'
    ]


if __name__ == "__main__":
    # Test the tracker
    logging.basicConfig(level=logging.INFO)

    tracker = OptionsFlowTracker()

    # Test single symbol
    test_symbol = 'AAPL'
    flow_data = tracker.fetch_options_flow(test_symbol)

    if flow_data:
        print(f"\nFlow Data for {test_symbol}:")
        print(f"  Call Volume: {flow_data.call_volume:,}")
        print(f"  Put Volume: {flow_data.put_volume:,}")
        print(f"  Net Premium Flow: ${flow_data.net_premium_flow:,.2f}")
        print(f"  Put/Call Ratio: {flow_data.put_call_ratio:.2f}")
        print(f"  Sentiment: {flow_data.flow_sentiment}")

        # Save to database
        tracker.save_flow_data(flow_data)

        # Analyze trend
        trend = tracker.analyze_flow_trend(test_symbol, days=7)
        print(f"\n7-Day Trend Analysis:")
        print(f"  Direction: {trend['trend_direction']}")
        print(f"  Net Flow: ${trend['net_flow']:,.2f}")
        print(f"  Dominant Sentiment: {trend['dominant_sentiment']}")
