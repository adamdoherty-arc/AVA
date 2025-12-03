"""
Unified Options Analyzer
Combines AI Options Agent (screening) + Comprehensive Strategy (deep analysis)
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

# Import existing analyzers
from src.ai_options_agent.options_analysis_agent import OptionsAnalysisAgent
from src.ai_options_agent.comprehensive_strategy_analyzer import ComprehensiveStrategyAnalyzer
from src.ai_options_agent.ai_options_db_manager import AIOptionsDBManager
from src.ai_options_agent.llm_manager import get_llm_manager

logger = logging.getLogger(__name__)


class UnifiedOptionsAnalyzer:
    """
    Unified analyzer combining:
    1. Screening capabilities (AI Options Agent)
    2. Strategy analysis (Comprehensive Strategy)
    3. Shared caching and optimization
    """

    def __init__(self, llm_manager=None):
        """
        Initialize unified analyzer

        Args:
            llm_manager: Optional LLM manager instance (creates new if None)
        """
        # Initialize LLM manager
        self.llm_manager = llm_manager or get_llm_manager()

        # Initialize database manager (shared by both analyzers)
        self.db_manager = AIOptionsDBManager()

        # Initialize screening agent (for batch analysis)
        self.screening_agent = OptionsAnalysisAgent(
            db_manager=self.db_manager,
            llm_manager=self.llm_manager
        )

        # Initialize strategy analyzer (for deep dive)
        self.strategy_analyzer = ComprehensiveStrategyAnalyzer(
            llm_manager=self.llm_manager
        )

        # Cache for results
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes

        logger.info("UnifiedOptionsAnalyzer initialized with both screening and strategy capabilities")

    def screen_opportunities(self,
                           source: str = "all",
                           watchlist_name: Optional[str] = None,
                           symbols: Optional[List[str]] = None,
                           dte_range: tuple = (20, 40),
                           delta_range: tuple = (-0.45, -0.15),
                           min_premium: float = 100.0,
                           limit: int = 200,
                           min_score: int = 50,
                           use_llm: bool = False,
                           llm_provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Screen for options opportunities (CSP focus)

        Args:
            source: 'all', 'watchlist', or 'symbols'
            watchlist_name: TradingView watchlist name (if source='watchlist')
            symbols: List of symbols (if source='symbols')
            dte_range: (min_dte, max_dte) tuple
            delta_range: (min_delta, max_delta) tuple
            min_premium: Minimum premium in dollars
            limit: Maximum results to analyze
            min_score: Minimum score to display (0-100)
            use_llm: Whether to use LLM for enhanced reasoning
            llm_provider: Specific LLM provider to use

        Returns:
            Dict with:
            - opportunities: List of analyzed opportunities
            - summary: Summary statistics
            - metadata: Execution metadata
        """
        start_time = time.time()

        # Run screening based on source
        if source == "watchlist" and watchlist_name:
            analyses = self.screening_agent.analyze_watchlist(
                watchlist_name=watchlist_name,
                dte_range=dte_range,
                delta_range=delta_range,
                min_premium=min_premium,
                limit=limit,
                use_llm=use_llm,
                llm_provider=llm_provider
            )
        elif source == "symbols" and symbols:
            # Analyze specific symbols
            analyses = []
            opportunities = self.db_manager.get_opportunities(
                symbols=symbols,
                dte_range=dte_range,
                delta_range=delta_range,
                min_premium=min_premium,
                limit=limit
            )
            for opp in opportunities:
                analysis = self.screening_agent.analyze_opportunity(
                    opp,
                    save_to_db=True,
                    use_llm=use_llm,
                    llm_provider=llm_provider
                )
                analyses.append(analysis)
        else:
            # Analyze all stocks
            analyses = self.screening_agent.analyze_all_stocks(
                dte_range=dte_range,
                delta_range=delta_range,
                min_premium=min_premium,
                limit=limit,
                use_llm=use_llm,
                llm_provider=llm_provider
            )

        # Filter by min score
        filtered_analyses = [a for a in analyses if a['final_score'] >= min_score]

        # Sort by score
        filtered_analyses.sort(key=lambda x: x['final_score'], reverse=True)

        # Calculate summary statistics
        summary = self._calculate_summary(filtered_analyses)

        # Execution metadata
        elapsed_time = time.time() - start_time
        metadata = {
            'execution_time': elapsed_time,
            'total_analyzed': len(analyses),
            'total_filtered': len(filtered_analyses),
            'filters': {
                'source': source,
                'watchlist_name': watchlist_name,
                'dte_range': dte_range,
                'delta_range': delta_range,
                'min_premium': min_premium,
                'min_score': min_score,
                'use_llm': use_llm
            },
            'timestamp': datetime.now().isoformat()
        }

        return {
            'opportunities': filtered_analyses,
            'summary': summary,
            'metadata': metadata
        }

    def analyze_stock_strategies(self,
                                symbol: str,
                                stock_data: Dict[str, Any],
                                options_data: Dict[str, Any],
                                use_multi_model: bool = True) -> Dict[str, Any]:
        """
        Deep dive strategy analysis for a single stock

        Args:
            symbol: Stock ticker
            stock_data: Dict with stock information
            options_data: Dict with options parameters
            use_multi_model: Use multi-model AI consensus

        Returns:
            Dict with:
            - symbol: Stock ticker
            - environment_analysis: Market environment
            - strategy_rankings: All 10 strategies ranked
            - top_recommendation: Best strategy
            - multi_model_consensus: AI consensus (if enabled)
        """
        # Check cache first
        cache_key = f"strategies_{symbol}_{options_data.get('strike_price')}_{options_data.get('dte')}"

        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if (datetime.now() - cache_entry['timestamp']).seconds < self._cache_ttl:
                logger.info(f"Using cached strategy analysis for {symbol}")
                return cache_entry['data']

        # Run comprehensive strategy analysis
        result = self.strategy_analyzer.analyze(
            symbol=symbol,
            stock_data=stock_data,
            options_data=options_data
        )

        # Cache the result
        self._cache[cache_key] = {
            'data': result,
            'timestamp': datetime.now()
        }

        return result

    def analyze_position(self,
                        position: Dict[str, Any],
                        use_multi_model: bool = True) -> Dict[str, Any]:
        """
        Analyze a current position to determine if it's still optimal

        Args:
            position: Position dict from PositionsManager
            use_multi_model: Use multi-model AI consensus

        Returns:
            Dict with:
            - current_position: Position details
            - position_analysis: Current Greeks, P&L, risks
            - strategy_rankings: All 10 strategies ranked
            - recommendation: Keep, adjust, or close
        """
        symbol = position.get('symbol')
        strike = position.get('strike')
        dte = position.get('dte')
        option_type = position.get('option_type')

        # Prepare stock data (fetch from database)
        from src.ai_options_agent.shared.data_fetchers import fetch_stock_info

        stock_info = fetch_stock_info(symbol)

        if not stock_info:
            return {
                'error': f'Could not fetch data for {symbol}',
                'current_position': position
            }

        stock_data = {
            'symbol': symbol,
            'current_price': stock_info.get('current_price', 0),
            'iv': position.get('iv', 0.35),  # Use position IV if available
            'price_52w_high': stock_info.get('high_52week', 0),
            'price_52w_low': stock_info.get('low_52week', 0),
            'market_cap': stock_info.get('market_cap', 0),
            'pe_ratio': stock_info.get('pe_ratio', 28.5),
            'sector': stock_info.get('sector', 'Unknown')
        }

        options_data = {
            'strike_price': strike,
            'dte': dte,
            'delta': position.get('delta', -0.30),
            'premium': position.get('entry_price', 0) * 100  # Convert to cents
        }

        # Analyze all strategies
        strategy_analysis = self.analyze_stock_strategies(
            symbol=symbol,
            stock_data=stock_data,
            options_data=options_data,
            use_multi_model=use_multi_model
        )

        # Determine recommendation
        recommendation = self._analyze_position_recommendation(
            position, strategy_analysis
        )

        return {
            'current_position': position,
            'position_analysis': {
                'pnl': position.get('pnl', 0),
                'pnl_pct': position.get('pnl_pct', 0),
                'delta': position.get('delta', 0),
                'theta': position.get('theta', 0),
                'gamma': position.get('gamma', 0),
                'vega': position.get('vega', 0),
                'dte': dte
            },
            'strategy_rankings': strategy_analysis.get('strategy_rankings', []),
            'environment_analysis': strategy_analysis.get('environment_analysis', {}),
            'recommendation': recommendation
        }

    def _calculate_summary(self, analyses: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics for screening results"""
        if not analyses:
            return {
                'total': 0,
                'strong_buys': 0,
                'buys': 0,
                'holds': 0,
                'avg_score': 0
            }

        strong_buys = len([a for a in analyses if a.get('recommendation') == 'STRONG_BUY'])
        buys = len([a for a in analyses if a.get('recommendation') == 'BUY'])
        holds = len([a for a in analyses if a.get('recommendation') == 'HOLD'])
        avg_score = sum(a['final_score'] for a in analyses) / len(analyses)

        return {
            'total': len(analyses),
            'strong_buys': strong_buys,
            'buys': buys,
            'holds': holds,
            'avg_score': avg_score
        }

    def _analyze_position_recommendation(self,
                                        position: Dict,
                                        strategy_analysis: Dict) -> str:
        """
        Analyze position and provide recommendation

        Returns:
            'KEEP', 'ADJUST', or 'CLOSE'
        """
        # Get current position type
        current_type = position.get('option_type', 'unknown')

        # Get top recommended strategy
        strategies = strategy_analysis.get('strategy_rankings', [])
        if not strategies:
            return 'KEEP'  # No analysis available

        top_strategy = strategies[0]

        # Check if current strategy is still in top 3
        current_in_top_3 = any(
            'cash-secured put' in s.get('name', '').lower()
            for s in strategies[:3]
        ) if current_type == 'put' else False

        # Get P&L
        pnl_pct = position.get('pnl_pct', 0)

        # Decision logic
        if current_in_top_3 and pnl_pct > -10:
            return 'KEEP'
        elif not current_in_top_3 and pnl_pct < -15:
            return 'CLOSE'
        else:
            return 'ADJUST'

    def clear_cache(self) -> None:
        """Clear analysis cache"""
        self._cache = {}
        logger.info("Analysis cache cleared")
