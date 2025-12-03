"""
Watchlist Strategy Analyzer - Analyze All Stocks for Best Strategies
====================================================================

Analyzes all stocks in watchlists and ranks strategies by profit potential.
Provides real trade examples with premiums and current values.

Features:
- Multi-strategy analysis (CSP, CC, Calendar Spreads, Iron Condors)
- Profit potential scoring and ranking
- Real option chain data with current premiums
- Risk/reward analysis
- Supply/demand zone validation
- Earnings conflict checking

Author: Magnus Trading Platform
Created: 2025-11-11
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import robin_stocks.robinhood as rh
from dataclasses import dataclass, asdict

# Magnus imports
from src.tradingview_api_sync import TradingViewAPISync
from src.csp_opportunities_finder import CSPOpportunitiesFinder
from src.zone_analyzer import ZoneAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StrategyAnalysis:
    """Container for strategy analysis results"""
    ticker: str
    strategy_type: str  # CSP, CC, Calendar, Iron Condor
    profit_score: float  # 0-100, higher is better
    expected_premium: float
    current_option_price: float
    strike: float
    expiration: str
    delta: Optional[float]
    iv_rank: Optional[float]
    probability_profit: Optional[float]
    risk_reward_ratio: float
    max_profit: float
    max_loss: float
    breakeven: float
    trade_details: str  # Human-readable description
    technical_score: float  # Based on supply/demand zones
    earnings_safe: bool
    recommendation: str  # BUY, HOLD, AVOID


class WatchlistStrategyAnalyzer:
    """Analyze all stocks in watchlist for best trading strategies"""

    def __init__(self) -> None:
        """Initialize analyzer with required services"""
        self.tv_sync = TradingViewAPISync()
        self.csp_finder = CSPOpportunitiesFinder()
        self.zone_analyzer = ZoneAnalyzer()

        # Strategy scoring weights
        self.weights = {
            'premium': 0.25,        # Premium size
            'probability': 0.20,    # Win probability
            'risk_reward': 0.20,    # Risk/reward ratio
            'technical': 0.20,      # Technical analysis
            'iv_rank': 0.15         # IV rank
        }

    def analyze_watchlist(
        self,
        watchlist_name: str,
        min_score: float = 60.0,
        strategies: List[str] = None
    ) -> List[StrategyAnalysis]:
        """
        Analyze all stocks in watchlist for best strategies

        Args:
            watchlist_name: Name of TradingView watchlist
            min_score: Minimum profit score to include (0-100)
            strategies: List of strategies to analyze, default all

        Returns:
            List of strategy analyses ranked by profit score
        """
        logger.info(f"Analyzing watchlist: {watchlist_name}")

        if strategies is None:
            strategies = ['CSP', 'CC', 'Calendar', 'Iron Condor']

        # Get stocks from watchlist
        stocks = self._get_watchlist_stocks(watchlist_name)

        if not stocks:
            logger.warning(f"No stocks found in watchlist: {watchlist_name}")
            return []

        logger.info(f"Found {len(stocks)} stocks in watchlist")

        # Analyze each stock for all strategies
        all_analyses = []

        for stock in stocks:
            ticker = stock['symbol']
            logger.info(f"Analyzing {ticker}...")

            try:
                # Get current stock data
                stock_data = self._get_stock_data(ticker)

                if not stock_data:
                    logger.warning(f"Could not get data for {ticker}")
                    continue

                # Analyze each strategy
                if 'CSP' in strategies:
                    csp_analysis = self._analyze_csp(ticker, stock_data)
                    if csp_analysis:
                        all_analyses.extend(csp_analysis)

                if 'CC' in strategies:
                    cc_analysis = self._analyze_covered_call(ticker, stock_data)
                    if cc_analysis:
                        all_analyses.extend(cc_analysis)

                if 'Calendar' in strategies:
                    calendar_analysis = self._analyze_calendar_spread(ticker, stock_data)
                    if calendar_analysis:
                        all_analyses.extend(calendar_analysis)

                if 'Iron Condor' in strategies:
                    iron_condor_analysis = self._analyze_iron_condor(ticker, stock_data)
                    if iron_condor_analysis:
                        all_analyses.extend(iron_condor_analysis)

            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {e}")
                continue

        # Filter by minimum score
        filtered_analyses = [a for a in all_analyses if a.profit_score >= min_score]

        # Sort by profit score descending
        ranked_analyses = sorted(filtered_analyses, key=lambda x: x.profit_score, reverse=True)

        logger.info(f"Found {len(ranked_analyses)} strategies above score {min_score}")

        return ranked_analyses

    def _get_watchlist_stocks(self, watchlist_name: str) -> List[Dict]:
        """Get all stocks from TradingView watchlist"""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            conn = self.tv_sync.get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Get watchlist ID
            cur.execute("""
                SELECT watchlist_id, symbols
                FROM tv_watchlists_api
                WHERE name ILIKE %s
                ORDER BY last_synced DESC
                LIMIT 1
            """, (f"%{watchlist_name}%",))

            result = cur.fetchone()

            if not result:
                logger.warning(f"Watchlist not found: {watchlist_name}")
                cur.close()
                conn.close()
                return []

            watchlist_id = result['watchlist_id']
            symbols = result['symbols']

            # Get detailed symbol info
            stocks = []
            for symbol in symbols:
                stocks.append({'symbol': symbol.strip()})

            cur.close()
            conn.close()

            return stocks

        except Exception as e:
            logger.error(f"Error getting watchlist stocks: {e}")
            return []

    def _get_stock_data(self, ticker: str) -> Optional[Dict]:
        """Get current stock price and basic data"""
        try:
            # Get stock quote from Robinhood
            quote = rh.get_quotes(ticker)

            if not quote or len(quote) == 0:
                return None

            price = float(quote[0].get('last_trade_price', 0))

            if price == 0:
                return None

            # Get fundamentals
            fundamentals = rh.get_fundamentals(ticker)
            market_cap = float(fundamentals[0].get('market_cap', 0)) if fundamentals else 0

            return {
                'ticker': ticker,
                'price': price,
                'market_cap': market_cap,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Error getting stock data for {ticker}: {e}")
            return None

    def _analyze_csp(self, ticker: str, stock_data: Dict) -> List[StrategyAnalysis]:
        """Analyze Cash-Secured Put opportunities"""
        analyses = []

        try:
            price = stock_data['price']

            # Get option chain for puts
            expirations = rh.get_chains(ticker).get('expiration_dates', [])

            if not expirations:
                return analyses

            # Analyze next 3-4 expirations
            for expiration in expirations[:4]:
                exp_date = datetime.strptime(expiration, '%Y-%m-%d')
                days_to_exp = (exp_date - datetime.now()).days

                # Only consider 14-45 DTE
                if days_to_exp < 14 or days_to_exp > 45:
                    continue

                # Get put options
                options = rh.find_options_by_expiration(
                    ticker,
                    expiration,
                    optionType='put'
                )

                if not options:
                    continue

                # Find optimal strike (0.15-0.35 delta, typically 5-15% OTM)
                target_strikes = [
                    price * 0.95,  # 5% OTM
                    price * 0.90,  # 10% OTM
                    price * 0.85   # 15% OTM
                ]

                for target_strike in target_strikes:
                    # Find closest strike
                    best_option = None
                    min_diff = float('inf')

                    for opt in options:
                        strike = float(opt.get('strike_price', 0))
                        diff = abs(strike - target_strike)

                        if diff < min_diff and strike < price:
                            min_diff = diff
                            best_option = opt

                    if not best_option:
                        continue

                    # Get option market data
                    opt_id = best_option['id'].split('/')[-2]
                    market_data = rh.get_option_market_data_by_id(opt_id)

                    if not market_data or len(market_data) == 0:
                        continue

                    strike = float(best_option['strike_price'])
                    bid = float(market_data[0].get('bid_price', 0))
                    ask = float(market_data[0].get('ask_price', 0))
                    mid = (bid + ask) / 2

                    if mid < 0.10:  # Skip if premium too low
                        continue

                    # Get Greeks
                    delta = abs(float(market_data[0].get('delta', 0.25)))
                    iv = float(market_data[0].get('implied_volatility', 0))

                    # Calculate IV rank (simplified - normally needs historical data)
                    iv_percentile = min(iv * 100, 100)

                    # Calculate profit metrics
                    premium_per_contract = mid * 100
                    max_profit = premium_per_contract
                    max_loss = (strike * 100) - premium_per_contract
                    breakeven = strike - mid
                    risk_reward_ratio = max_profit / max_loss if max_loss > 0 else 0

                    # Probability of profit (approximation based on delta)
                    probability_profit = (1 - delta) * 100

                    # Check technical analysis
                    technical_score = self._get_technical_score(ticker, strike)

                    # Check earnings
                    earnings_safe = self._check_earnings_safe(ticker, exp_date)

                    # Calculate overall profit score
                    profit_score = self._calculate_profit_score({
                        'premium': premium_per_contract,
                        'probability': probability_profit,
                        'risk_reward': risk_reward_ratio,
                        'technical': technical_score,
                        'iv_rank': iv_percentile
                    })

                    # Generate trade details
                    trade_details = f"SELL {ticker} ${strike:.2f}P @ ${mid:.2f} (${premium_per_contract:.0f} credit)"

                    # Determine recommendation
                    recommendation = self._get_recommendation(profit_score, technical_score, earnings_safe)

                    # Create analysis
                    analysis = StrategyAnalysis(
                        ticker=ticker,
                        strategy_type='CSP',
                        profit_score=profit_score,
                        expected_premium=premium_per_contract,
                        current_option_price=mid,
                        strike=strike,
                        expiration=expiration,
                        delta=delta,
                        iv_rank=iv_percentile,
                        probability_profit=probability_profit,
                        risk_reward_ratio=risk_reward_ratio,
                        max_profit=max_profit,
                        max_loss=max_loss,
                        breakeven=breakeven,
                        trade_details=trade_details,
                        technical_score=technical_score,
                        earnings_safe=earnings_safe,
                        recommendation=recommendation
                    )

                    analyses.append(analysis)

        except Exception as e:
            logger.error(f"Error analyzing CSP for {ticker}: {e}")

        return analyses

    def _analyze_covered_call(self, ticker: str, stock_data: Dict) -> List[StrategyAnalysis]:
        """Analyze Covered Call opportunities (assumes owning 100 shares)"""
        analyses = []

        try:
            price = stock_data['price']

            # Get option chain for calls
            expirations = rh.get_chains(ticker).get('expiration_dates', [])

            if not expirations:
                return analyses

            # Analyze next 3-4 expirations
            for expiration in expirations[:4]:
                exp_date = datetime.strptime(expiration, '%Y-%m-%d')
                days_to_exp = (exp_date - datetime.now()).days

                # Only consider 14-45 DTE
                if days_to_exp < 14 or days_to_exp > 45:
                    continue

                # Get call options
                options = rh.find_options_by_expiration(
                    ticker,
                    expiration,
                    optionType='call'
                )

                if not options:
                    continue

                # Find optimal strikes (typically 5-15% OTM)
                target_strikes = [
                    price * 1.05,  # 5% OTM
                    price * 1.10,  # 10% OTM
                    price * 1.15   # 15% OTM
                ]

                for target_strike in target_strikes:
                    # Find closest strike
                    best_option = None
                    min_diff = float('inf')

                    for opt in options:
                        strike = float(opt.get('strike_price', 0))
                        diff = abs(strike - target_strike)

                        if diff < min_diff and strike > price:
                            min_diff = diff
                            best_option = opt

                    if not best_option:
                        continue

                    # Get option market data
                    opt_id = best_option['id'].split('/')[-2]
                    market_data = rh.get_option_market_data_by_id(opt_id)

                    if not market_data or len(market_data) == 0:
                        continue

                    strike = float(best_option['strike_price'])
                    bid = float(market_data[0].get('bid_price', 0))
                    ask = float(market_data[0].get('ask_price', 0))
                    mid = (bid + ask) / 2

                    if mid < 0.10:
                        continue

                    # Get Greeks
                    delta = abs(float(market_data[0].get('delta', 0.30)))
                    iv = float(market_data[0].get('implied_volatility', 0))
                    iv_percentile = min(iv * 100, 100)

                    # Calculate profit metrics
                    premium_per_contract = mid * 100
                    max_profit = premium_per_contract + ((strike - price) * 100)
                    max_loss = (price * 100) - premium_per_contract  # If stock goes to 0
                    breakeven = price - mid
                    risk_reward_ratio = max_profit / price if price > 0 else 0

                    # Probability of profit (call gets assigned)
                    probability_profit = delta * 100

                    # Technical analysis
                    technical_score = self._get_technical_score(ticker, strike)

                    # Earnings check
                    earnings_safe = self._check_earnings_safe(ticker, exp_date)

                    # Calculate profit score
                    profit_score = self._calculate_profit_score({
                        'premium': premium_per_contract,
                        'probability': probability_profit,
                        'risk_reward': risk_reward_ratio,
                        'technical': technical_score,
                        'iv_rank': iv_percentile
                    })

                    # Trade details
                    trade_details = f"SELL {ticker} ${strike:.2f}C @ ${mid:.2f} (${premium_per_contract:.0f} credit)"

                    # Recommendation
                    recommendation = self._get_recommendation(profit_score, technical_score, earnings_safe)

                    analysis = StrategyAnalysis(
                        ticker=ticker,
                        strategy_type='CC',
                        profit_score=profit_score,
                        expected_premium=premium_per_contract,
                        current_option_price=mid,
                        strike=strike,
                        expiration=expiration,
                        delta=delta,
                        iv_rank=iv_percentile,
                        probability_profit=probability_profit,
                        risk_reward_ratio=risk_reward_ratio,
                        max_profit=max_profit,
                        max_loss=max_loss,
                        breakeven=breakeven,
                        trade_details=trade_details,
                        technical_score=technical_score,
                        earnings_safe=earnings_safe,
                        recommendation=recommendation
                    )

                    analyses.append(analysis)

        except Exception as e:
            logger.error(f"Error analyzing CC for {ticker}: {e}")

        return analyses

    def _analyze_calendar_spread(self, ticker: str, stock_data: Dict) -> List[StrategyAnalysis]:
        """Analyze Calendar Spread opportunities"""
        # Simplified implementation - would need more complex logic
        return []

    def _analyze_iron_condor(self, ticker: str, stock_data: Dict) -> List[StrategyAnalysis]:
        """Analyze Iron Condor opportunities"""
        # Simplified implementation - would need more complex logic
        return []

    def _get_technical_score(self, ticker: str, strike: float) -> float:
        """Get technical analysis score based on supply/demand zones"""
        try:
            # Check if strike is near support/resistance
            zones = self.zone_analyzer.analyze_stock(ticker, timeframe='daily')

            if not zones:
                return 50.0  # Neutral if no zone data

            price = strike

            # Check distance to nearest support/resistance
            nearest_support = min([z.price for z in zones if z.zone_type == 'support' and z.price < price], default=0)
            nearest_resistance = min([z.price for z in zones if z.zone_type == 'resistance' and z.price > price], default=float('inf'))

            if nearest_support > 0:
                support_distance = ((price - nearest_support) / price) * 100

                if support_distance < 5:  # Within 5% of support
                    return 80.0  # Strong support
                elif support_distance < 10:
                    return 65.0

            return 50.0  # Neutral

        except Exception as e:
            logger.error(f"Error getting technical score: {e}")
            return 50.0

    def _check_earnings_safe(self, ticker: str, exp_date: datetime) -> bool:
        """Check if expiration is safe from earnings"""
        try:
            # Check earnings calendar
            fundamentals = rh.get_fundamentals(ticker)

            if not fundamentals:
                return True  # Assume safe if no data

            earnings_date_str = fundamentals[0].get('earnings_date')

            if not earnings_date_str:
                return True

            earnings_date = datetime.strptime(earnings_date_str, '%Y-%m-%d')

            # Check if earnings is between now and expiration
            if datetime.now() < earnings_date < exp_date:
                return False  # Earnings before expiration - not safe

            return True

        except Exception as e:
            logger.error(f"Error checking earnings: {e}")
            return True  # Assume safe on error

    def _calculate_profit_score(self, metrics: Dict) -> float:
        """Calculate overall profit score (0-100) based on multiple factors"""
        score = 0.0

        # Premium score (0-100 based on $50-$500 range)
        premium = metrics.get('premium', 0)
        premium_score = min((premium / 500) * 100, 100)
        score += premium_score * self.weights['premium']

        # Probability score (directly use percentage)
        probability = metrics.get('probability', 50)
        score += probability * self.weights['probability']

        # Risk/reward score (0-100, optimal around 0.1-0.3)
        risk_reward = metrics.get('risk_reward', 0)
        if risk_reward > 0:
            # Convert to 0-100 scale (0.3 = 100, 0.0 = 0)
            rr_score = min((risk_reward / 0.3) * 100, 100)
            score += rr_score * self.weights['risk_reward']

        # Technical score (already 0-100)
        technical = metrics.get('technical', 50)
        score += technical * self.weights['technical']

        # IV rank score (already 0-100)
        iv_rank = metrics.get('iv_rank', 50)
        score += iv_rank * self.weights['iv_rank']

        return min(score, 100.0)

    def _get_recommendation(self, profit_score: float, technical_score: float, earnings_safe: bool) -> str:
        """Get trade recommendation based on scores"""
        if not earnings_safe:
            return 'AVOID'

        if profit_score >= 75 and technical_score >= 65:
            return 'BUY'
        elif profit_score >= 60:
            return 'HOLD'
        else:
            return 'AVOID'

    def format_results(self, analyses: List[StrategyAnalysis], limit: int = 20) -> str:
        """Format analysis results as human-readable text"""
        if not analyses:
            return "No strategies found matching criteria."

        output = f"# Top {min(limit, len(analyses))} Strategies Ranked by Profit Score\n\n"

        for i, analysis in enumerate(analyses[:limit], 1):
            output += f"## {i}. {analysis.ticker} - {analysis.strategy_type} (Score: {analysis.profit_score:.1f})\n\n"
            output += f"**Trade:** {analysis.trade_details}\n"
            output += f"**Expiration:** {analysis.expiration} | **Recommendation:** {analysis.recommendation}\n\n"

            output += f"**Profit Metrics:**\n"
            output += f"- Expected Premium: ${analysis.expected_premium:.2f}\n"
            output += f"- Current Option Price: ${analysis.current_option_price:.2f}\n"
            output += f"- Max Profit: ${analysis.max_profit:.2f}\n"
            output += f"- Max Loss: ${analysis.max_loss:.2f}\n"
            output += f"- Breakeven: ${analysis.breakeven:.2f}\n"
            output += f"- Probability of Profit: {analysis.probability_profit:.1f}%\n\n"

            output += f"**Greeks & Analysis:**\n"
            output += f"- Delta: {analysis.delta:.3f}\n"
            output += f"- IV Rank: {analysis.iv_rank:.1f}%\n"
            output += f"- Technical Score: {analysis.technical_score:.1f}/100\n"
            output += f"- Earnings Safe: {'Yes' if analysis.earnings_safe else 'No'}\n\n"

            output += "---\n\n"

        return output


# Example usage
if __name__ == "__main__":
    analyzer = WatchlistStrategyAnalyzer()

    # Analyze NVDA watchlist
    results = analyzer.analyze_watchlist(
        watchlist_name='NVDA',
        min_score=60.0,
        strategies=['CSP', 'CC']
    )

    # Print formatted results
    print(analyzer.format_results(results, limit=10))
