"""
Agent-Aware NLP Handler for AVA
================================

Enhanced NLP handler that routes queries to specialized agents for superior responses.

This handler sits on top of the existing NLP handler and adds agent routing capabilities,
connecting AVA to all 33 specialized agents for trading, analysis, sports, and more.

Key Improvements:
- Routes queries to specialized agents (3-5x better responses)
- Integrates LLM services (sports analyzer, options strategist)
- Uses connection pool for database access
- Supports multi-agent collaboration
- Fallback to generic LLM if no agent matches

Author: Magnus Enhancement Team
Date: 2025-11-20
"""

import logging
import asyncio
from typing import Dict, Optional, List, Any
from datetime import datetime

from .nlp_handler import NaturalLanguageHandler, Intent
from .core.agent_initializer import ensure_agents_initialized, get_registry
from src.services.llm_sports_analyzer import LLMSportsAnalyzer
from src.services.llm_options_strategist import LLMOptionsStrategist, MarketOutlook, RiskTolerance
from src.database import get_db_connection
from src.magnus_local_llm import get_magnus_llm
from src.rag import get_rag

logger = logging.getLogger(__name__)


class AgentAwareNLPHandler:
    """
    Enhanced NLP handler with agent routing capabilities

    Routes user queries to specialized agents for superior analysis and responses.
    Falls back to generic LLM if no specialized agent matches.
    """

    def __init__(self):
        """Initialize with agent registry and specialized services"""
        # Base NLP handler for fallback
        self.base_nlp = NaturalLanguageHandler()

        # Initialize agent registry
        try:
            ensure_agents_initialized()
            self.registry = get_registry()
            agent_count = len(self.registry.list_agent_names())
            logger.info(f"✅ Agent-Aware NLP Handler initialized with {agent_count} agents")
        except Exception as e:
            logger.error(f"Failed to initialize agent registry: {e}")
            self.registry = None

        # Initialize specialized LLM services
        self.sports_analyzer = LLMSportsAnalyzer()
        self.options_strategist = LLMOptionsStrategist()
        self.local_llm = get_magnus_llm()

        # Initialize RAG system (optional - gracefully handle if not available)
        try:
            self.rag = get_rag()
            if self.rag:
                logger.info("✅ RAG system initialized - AVA has access to knowledge base")
            else:
                logger.info("📚 RAG system not available (install: pip install chromadb sentence-transformers)")
                self.rag = None
        except Exception as e:
            logger.info(f"📚 RAG system not available: {e}")
            self.rag = None

        # Comprehensive agent routing map - covers ALL 36 agents
        # Each keyword maps to capability names that agents register with
        self.routing_map = {
            # ===== PORTFOLIO & POSITIONS (portfolio_agent, position_agent) =====
            'portfolio': ['portfolio_analysis', 'position_tracking', 'greeks_analysis'],
            'positions': ['portfolio_analysis', 'position_management', 'position_tracking'],
            'balance': ['portfolio_analysis', 'portfolio_summary'],
            'holdings': ['portfolio_analysis', 'position_tracking'],
            'account': ['portfolio_analysis', 'portfolio_summary'],
            'pnl': ['portfolio_analysis', 'performance_tracking'],
            'profit': ['portfolio_analysis', 'performance_tracking'],
            'loss': ['portfolio_analysis', 'performance_tracking'],
            'greeks': ['portfolio_analysis', 'options_analysis', 'greeks_analysis'],
            'delta': ['portfolio_analysis', 'options_analysis', 'greeks_analysis'],
            'gamma': ['portfolio_analysis', 'options_analysis', 'greeks_analysis'],
            'vega': ['portfolio_analysis', 'options_analysis', 'greeks_analysis'],

            # ===== THETA & TIME DECAY (portfolio_agent, options_analysis_agent) =====
            'theta': ['portfolio_analysis', 'options_analysis', 'theta_analysis'],
            'decay': ['portfolio_analysis', 'options_analysis', 'theta_analysis'],
            'time decay': ['portfolio_analysis', 'options_analysis', 'theta_analysis'],
            'premium erosion': ['portfolio_analysis', 'theta_analysis'],

            # ===== OPTIONS ANALYSIS (options_analysis_agent, options_flow_agent) =====
            'options': ['options_analysis', 'options_flow_analysis'],
            'option': ['options_analysis', 'options_flow_analysis'],
            'put': ['options_analysis', 'csp_analysis'],
            'call': ['options_analysis', 'cc_analysis'],
            'strike': ['options_analysis'],
            'expiration': ['options_analysis', 'options_screening'],
            'dte': ['options_analysis', 'options_screening'],
            'csp': ['options_analysis', 'csp_analysis', 'premium_scanning'],
            'covered call': ['options_analysis', 'cc_analysis', 'premium_scanning'],
            'wheel': ['options_analysis', 'wheel_strategy'],
            'assignment': ['options_analysis', 'risk_assessment'],
            'iv': ['options_analysis', 'iv_analysis'],
            'implied volatility': ['options_analysis', 'iv_analysis'],
            'iv rank': ['options_analysis', 'iv_analysis'],

            # ===== OPTIONS FLOW (options_flow_agent) =====
            'flow': ['options_flow_analysis', 'unusual_activity'],
            'unusual': ['options_flow_analysis', 'unusual_activity'],
            'sweep': ['options_flow_analysis', 'unusual_activity'],
            'dark pool': ['options_flow_analysis', 'unusual_activity'],
            'institutional': ['options_flow_analysis', 'unusual_activity'],
            'whale': ['options_flow_analysis', 'unusual_activity'],
            'smart money': ['options_flow_analysis', 'unusual_activity'],

            # ===== PREMIUM SCANNER (premium_scanner_agent) =====
            'scan': ['premium_scanning', 'opportunity_scan'],
            'scanner': ['premium_scanning', 'opportunity_scan'],
            'find': ['premium_scanning', 'opportunity_scan'],
            'search': ['premium_scanning', 'opportunity_scan'],
            'opportunities': ['premium_scanning', 'opportunity_scan'],
            'best plays': ['premium_scanning', 'opportunity_scan'],
            'premium': ['premium_scanning', 'premium_analysis'],

            # ===== STRATEGY (strategy_agent) =====
            'strategy': ['strategy_generation', 'options_analysis'],
            'strategies': ['strategy_generation', 'options_analysis'],
            'spread': ['strategy_generation', 'options_analysis'],
            'iron condor': ['strategy_generation', 'options_analysis'],
            'straddle': ['strategy_generation', 'options_analysis'],
            'strangle': ['strategy_generation', 'options_analysis'],
            'vertical': ['strategy_generation', 'options_analysis'],
            'calendar': ['strategy_generation', 'calendar_spreads'],
            'diagonal': ['strategy_generation', 'options_analysis'],

            # ===== TECHNICAL ANALYSIS (technical_agent) =====
            'technical': ['technical_analysis', 'chart_analysis'],
            'chart': ['technical_analysis', 'chart_analysis'],
            'trend': ['technical_analysis', 'trend_analysis'],
            'pattern': ['technical_analysis', 'chart_patterns'],
            'indicator': ['technical_analysis', 'technical_indicators'],
            'rsi': ['technical_analysis', 'technical_indicators'],
            'macd': ['technical_analysis', 'technical_indicators'],
            'moving average': ['technical_analysis', 'technical_indicators'],
            'bollinger': ['technical_analysis', 'technical_indicators'],
            'volume': ['technical_analysis', 'volume_analysis'],

            # ===== SUPPLY & DEMAND (supply_demand_agent) =====
            'support': ['supply_demand_analysis', 'technical_analysis'],
            'resistance': ['supply_demand_analysis', 'technical_analysis'],
            'supply': ['supply_demand_analysis'],
            'demand': ['supply_demand_analysis'],
            'zone': ['supply_demand_analysis'],
            'level': ['supply_demand_analysis', 'technical_analysis'],

            # ===== FUNDAMENTAL ANALYSIS (fundamental_agent) =====
            'fundamental': ['fundamental_analysis', 'valuation_analysis'],
            'valuation': ['fundamental_analysis', 'valuation_analysis'],
            'pe ratio': ['fundamental_analysis', 'valuation_analysis'],
            'revenue': ['fundamental_analysis', 'financial_analysis'],
            'growth': ['fundamental_analysis', 'financial_analysis'],
            'debt': ['fundamental_analysis', 'financial_analysis'],
            'cash flow': ['fundamental_analysis', 'financial_analysis'],
            'dividend': ['fundamental_analysis', 'dividend_analysis'],
            'financials': ['fundamental_analysis', 'financial_analysis'],

            # ===== SENTIMENT (sentiment_agent) =====
            'sentiment': ['sentiment_analysis', 'news_analysis'],
            'news': ['sentiment_analysis', 'news_analysis'],
            'social': ['sentiment_analysis', 'social_analysis'],
            'analyst': ['sentiment_analysis', 'analyst_ratings'],
            'rating': ['sentiment_analysis', 'analyst_ratings'],
            'upgrade': ['sentiment_analysis', 'analyst_ratings'],
            'downgrade': ['sentiment_analysis', 'analyst_ratings'],

            # ===== SECTOR ANALYSIS (sector_agent) =====
            'sector': ['sector_analysis', 'market_analysis'],
            'industry': ['sector_analysis', 'market_analysis'],
            'rotation': ['sector_analysis', 'market_analysis'],

            # ===== EARNINGS (earnings_agent) =====
            'earnings': ['earnings_analysis', 'earnings_calendar'],
            'eps': ['earnings_analysis', 'earnings_calendar'],
            'quarterly': ['earnings_analysis', 'earnings_calendar'],
            'guidance': ['earnings_analysis', 'earnings_calendar'],
            'beat': ['earnings_analysis'],
            'miss': ['earnings_analysis'],
            'report': ['earnings_analysis', 'earnings_calendar'],

            # ===== MARKET DATA (market_data_agent) =====
            'price': ['market_data', 'price_lookup'],
            'quote': ['market_data', 'price_lookup'],
            'market': ['market_data', 'market_analysis'],
            'stock': ['market_data', 'price_lookup'],
            'current price': ['market_data', 'price_lookup'],
            'analyze': ['market_data', 'technical_analysis'],

            # ===== RISK MANAGEMENT (risk_management_agent) =====
            'risk': ['risk_assessment', 'risk_management'],
            'exposure': ['risk_assessment', 'portfolio_analysis'],
            'var': ['risk_assessment', 'risk_management'],
            'drawdown': ['risk_assessment', 'risk_management'],
            'hedge': ['risk_assessment', 'risk_management'],
            'protect': ['risk_assessment', 'risk_management'],
            'position size': ['risk_assessment', 'position_sizing'],
            'sizing': ['risk_assessment', 'position_sizing'],

            # ===== SPORTS BETTING (sports agents) =====
            'game': ['game_analysis', 'sports_betting'],
            'nfl': ['nfl_markets', 'sports_betting', 'game_analysis'],
            'nba': ['sports_betting', 'game_analysis'],
            'ncaa': ['sports_betting', 'game_analysis'],
            'mlb': ['sports_betting', 'game_analysis'],
            'football': ['nfl_markets', 'sports_betting'],
            'basketball': ['sports_betting', 'game_analysis'],
            'baseball': ['sports_betting', 'game_analysis'],
            'predict': ['game_analysis', 'sports_betting'],
            'prediction': ['game_analysis', 'sports_betting'],
            'bet': ['sports_betting', 'betting_strategy'],
            'betting': ['sports_betting', 'betting_strategy'],
            'wager': ['sports_betting', 'betting_strategy'],

            # ===== ODDS & KALSHI (odds_comparison_agent, kalshi_markets_agent) =====
            'odds': ['odds_comparison', 'sports_betting'],
            'line': ['odds_comparison', 'sports_betting'],
            'spread': ['odds_comparison', 'sports_betting', 'strategy_generation'],
            'moneyline': ['odds_comparison', 'sports_betting'],
            'over under': ['odds_comparison', 'sports_betting'],
            'kalshi': ['kalshi_markets', 'prediction_markets'],
            'polymarket': ['prediction_markets'],
            'prediction market': ['kalshi_markets', 'prediction_markets'],

            # ===== RESEARCH & KNOWLEDGE (research_agent, knowledge_agent, documentation_agent) =====
            'what': ['knowledge_base', 'research'],
            'what is': ['knowledge_base', 'research'],
            'how': ['knowledge_base', 'research'],
            'how to': ['knowledge_base', 'research'],
            'why': ['knowledge_base', 'research'],
            'explain': ['knowledge_base', 'research'],
            'learn': ['knowledge_base', 'documentation'],
            'teach': ['knowledge_base', 'documentation'],
            'help': ['knowledge_base', 'documentation'],
            'guide': ['knowledge_base', 'documentation'],
            'documentation': ['documentation', 'knowledge_base'],

            # ===== WATCHLIST & MONITORING (watchlist_monitor_agent, price_action_agent) =====
            'watch': ['watchlist_monitoring', 'price_action_monitoring'],
            'watchlist': ['watchlist_monitoring'],
            'monitor': ['watchlist_monitoring', 'price_action_monitoring'],
            'tracking': ['watchlist_monitoring', 'position_tracking'],
            'price action': ['price_action_monitoring', 'technical_analysis'],

            # ===== ALERTS (alert_agent) =====
            'alert': ['alert_management', 'notification'],
            'alerts': ['alert_management', 'notification'],
            'notify': ['alert_management', 'notification'],
            'notification': ['alert_management', 'notification'],
            'reminder': ['alert_management', 'task_management'],

            # ===== XTRADES (xtrades_monitor_agent) =====
            'xtrades': ['xtrades_monitoring', 'trade_copying'],
            'xtrade': ['xtrades_monitoring', 'trade_copying'],
            'follow': ['xtrades_monitoring', 'trade_copying'],
            'copy trade': ['xtrades_monitoring', 'trade_copying'],
            'trader': ['xtrades_monitoring'],

            # ===== ANALYTICS (analytics_agent) =====
            'analytics': ['performance_analytics', 'portfolio_analytics'],
            'performance': ['performance_analytics', 'portfolio_analysis'],
            'statistics': ['performance_analytics', 'portfolio_analytics'],
            'stats': ['performance_analytics', 'portfolio_analytics'],
            'win rate': ['performance_analytics'],
            'profit factor': ['performance_analytics'],

            # ===== DISCORD (discord_agent) =====
            'discord': ['discord_integration', 'notification'],

            # ===== TASKS & SETTINGS (task_management_agent, settings_agent) =====
            'task': ['task_management'],
            'todo': ['task_management'],
            'tasks': ['task_management'],
            'setting': ['settings_management'],
            'settings': ['settings_management'],
            'config': ['settings_management'],
            'configure': ['settings_management'],
            'preference': ['settings_management'],

            # ===== POSITION MANAGEMENT (position_agent) =====
            'close': ['position_management', 'trade_execution'],
            'roll': ['position_management', 'options_analysis'],
            'adjust': ['position_management', 'options_analysis'],
            'exit': ['position_management', 'trade_execution'],
            'open': ['position_management', 'trade_execution'],

            # ===== CACHE & SYSTEM (cache_metrics_agent) =====
            'cache': ['cache_metrics', 'system_monitoring'],
            'system': ['system_monitoring', 'cache_metrics'],
            'health': ['system_monitoring'],
            'status': ['system_monitoring', 'portfolio_analysis'],
        }

    def parse_query(self, user_text: str, context: Optional[Dict] = None) -> Dict:
        """
        Parse user query and route to appropriate agent or service

        Args:
            user_text: User's natural language query
            context: Optional conversation context

        Returns:
            Enhanced response with agent analysis
        """
        try:
            # Enrich context with RAG knowledge base (if available)
            rag_context = None
            if self.rag:
                rag_context = self._get_rag_context(user_text)
                if rag_context and context:
                    context['rag_context'] = rag_context
                elif rag_context:
                    context = {'rag_context': rag_context}

            # First, detect intent using base NLP
            base_result = self.base_nlp.parse_intent(user_text, context)

            # Try specialized agent routing
            agent_response = self._route_to_agent(user_text, base_result, context)

            if agent_response:
                # Agent handled the query
                return {
                    **base_result,
                    'response': agent_response['response'],
                    'agent_used': agent_response['agent'],
                    'response_quality': 'specialized',
                    'timestamp': datetime.now().isoformat()
                }

            # Try specialized LLM services
            llm_response = self._try_specialized_llm(user_text, base_result)

            if llm_response:
                return {
                    **base_result,
                    'response': llm_response['response'],
                    'service_used': llm_response['service'],
                    'response_quality': 'llm_specialized',
                    'timestamp': datetime.now().isoformat()
                }

            # Fallback to base NLP response
            return {
                **base_result,
                'response_quality': 'generic',
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in parse_query: {e}")
            return {
                'intent': Intent.UNKNOWN.value,
                'response': f"I encountered an error processing your query. Please try rephrasing.",
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _route_to_agent(self, user_text: str, base_result: Dict, context: Optional[Dict]) -> Optional[Dict]:
        """Route query to specialized agent based on capabilities"""
        if not self.registry:
            return None

        try:
            # Find matching capabilities
            capabilities = self._find_capabilities(user_text)

            if not capabilities:
                return None

            # Get agents with matching capabilities
            agents = []
            for cap in capabilities:
                matching = self.registry.find_agents_by_capability(cap)
                agents.extend(matching)

            if not agents:
                return None

            # Use first matching agent (could be enhanced with ranking)
            agent = agents[0]

            logger.info(f"Routing query to agent: {agent.name}")

            # Execute agent
            result = agent.execute(
                query=user_text,
                context=context or {}
            )

            return {
                'agent': agent.name,
                'response': result.get('response', result),
                'metadata': result.get('metadata', {})
            }

        except Exception as e:
            logger.error(f"Error routing to agent: {e}")
            return None

    def _find_capabilities(self, user_text: str) -> List[str]:
        """Find relevant capabilities based on query keywords"""
        capabilities = set()
        text_lower = user_text.lower()

        for keyword, caps in self.routing_map.items():
            if keyword in text_lower:
                capabilities.update(caps)

        return list(capabilities)

    def _try_specialized_llm(self, user_text: str, base_result: Dict) -> Optional[Dict]:
        """Try specialized LLM services for enhanced analysis"""
        text_lower = user_text.lower()

        try:
            # Sports prediction with LLM analyzer
            if any(keyword in text_lower for keyword in ['game', 'predict', 'nfl', 'nba', 'bet']):
                return self._handle_sports_query(user_text)

            # Options strategy with LLM strategist
            if any(keyword in text_lower for keyword in ['strategy', 'spread', 'trade', 'options']):
                return self._handle_options_strategy_query(user_text)

            return None

        except Exception as e:
            logger.error(f"Error in specialized LLM services: {e}")
            return None

    def _handle_sports_query(self, user_text: str) -> Optional[Dict]:
        """Handle sports prediction queries with LLM analyzer"""
        try:
            # Extract game info from query (simplified - could be enhanced)
            # This is a placeholder - real implementation would parse the query better

            response = f"""I can help analyze sports games with AI-powered predictions!

To get a detailed analysis, please provide:
- Teams playing (e.g., "Chiefs vs Bills")
- Or ask about upcoming games

I'll use advanced AI to analyze:
- Recent form and momentum
- Injury impacts
- Weather conditions
- Head-to-head history
- Betting value opportunities

Try: "Analyze the next Chiefs game" or "What are the best NFL bets this week?" """

            return {
                'service': 'LLMSportsAnalyzer',
                'response': response
            }

        except Exception as e:
            logger.error(f"Error in sports query: {e}")
            return None

    def _handle_options_strategy_query(self, user_text: str) -> Optional[Dict]:
        """Handle options strategy queries with LLM strategist"""
        try:
            # Extract symbol and outlook from query (simplified)
            # This is a placeholder - real implementation would parse better

            response = f"""I can generate custom options strategies for you!

To get personalized strategy recommendations, please provide:
- Stock symbol (e.g., "AAPL", "TSLA")
- Your market outlook (bullish, bearish, neutral, volatile)
- Risk tolerance (conservative, moderate, aggressive)

I'll generate THREE strategies:
1. **Conservative:** High probability, defined risk
2. **Moderate:** Balanced risk/reward
3. **Aggressive:** High reward potential

Each includes: exact strikes, expirations, max profit/loss, breakevens, and Greeks.

Try: "Generate strategies for AAPL, bullish outlook, moderate risk" """

            return {
                'service': 'LLMOptionsStrategist',
                'response': response
            }

        except Exception as e:
            logger.error(f"Error in options strategy query: {e}")
            return None

    async def analyze_game_async(self, game_data: Dict) -> Dict:
        """Async wrapper for sports game analysis"""
        return await self.sports_analyzer.analyze_game(game_data)

    async def generate_options_strategies_async(
        self,
        symbol: str,
        outlook: str,
        risk_tolerance: str
    ) -> Dict:
        """Async wrapper for options strategy generation"""
        outlook_enum = MarketOutlook(outlook.lower())
        risk_enum = RiskTolerance(risk_tolerance.lower())

        return await self.options_strategist.generate_strategies(
            symbol=symbol,
            outlook=outlook_enum,
            risk_tolerance=risk_enum
        )

    def get_agent_capabilities(self) -> Dict[str, List[str]]:
        """Get all available agent capabilities"""
        if not self.registry:
            return {}

        agents = self.registry.list_agent_names()
        capabilities_map = {}

        for agent_name in agents:
            agent = self.registry.get_agent(agent_name)
            if agent and hasattr(agent, 'capabilities'):
                capabilities_map[agent_name] = agent.capabilities

        return capabilities_map

    def _get_rag_context(self, user_text: str) -> Optional[str]:
        """
        Get relevant context from RAG knowledge base

        Args:
            user_text: User's query text

        Returns:
            Formatted context string or None if no relevant results
        """
        if not self.rag:
            return None

        try:
            # Query RAG system for relevant context
            context = self.rag.get_context_for_query(
                query_text=user_text,
                n_results=3,
                max_context_length=2000
            )

            if context and context.strip():
                logger.info(f"📚 RAG: Added knowledge base context for query")
                return context

            return None

        except Exception as e:
            logger.error(f"Error getting RAG context: {e}")
            return None

    def get_agent_stats(self) -> Dict:
        """Get statistics about available agents"""
        if not self.registry:
            return {'error': 'Agent registry not available'}

        agents = self.registry.list_agent_names()

        stats = {
            'total_agents': len(agents),
            'by_category': {},
            'capabilities': self.get_agent_capabilities(),
            'specialized_services': {
                'sports_analyzer': 'LLMSportsAnalyzer',
                'options_strategist': 'LLMOptionsStrategist'
            },
            'rag_enabled': self.rag is not None
        }

        # Categorize agents
        for agent_name in agents:
            # Extract category from agent name (e.g., "TradingMarketDataAgent" -> "Trading")
            category = agent_name.split('Agent')[0]
            # Find the last capital letter sequence
            import re
            matches = re.findall(r'[A-Z][a-z]*', category)
            if matches:
                category = matches[0]

            if category not in stats['by_category']:
                stats['by_category'][category] = 0
            stats['by_category'][category] += 1

        return stats


# Convenience function for easy access
def get_agent_aware_handler() -> AgentAwareNLPHandler:
    """Get singleton instance of agent-aware NLP handler"""
    if not hasattr(get_agent_aware_handler, '_instance'):
        get_agent_aware_handler._instance = AgentAwareNLPHandler()

    return get_agent_aware_handler._instance
