"""
AVA Tools - LangChain tool definitions
"""

from langchain_core.tools import tool
from typing import Dict, Any, Optional
from contextlib import contextmanager
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import logging

logger = logging.getLogger(__name__)


@contextmanager
def _safe_db_connection():
    """Context manager for safe database connections - prevents leaks."""
    conn = None
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'magnus'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )
        yield conn
    finally:
        if conn:
            conn.close()


@tool
def query_database_tool(query: str) -> str:
    """
    Execute a SQL query on the Magnus database.
    Use this to get information about tasks, positions, watchlists, etc.

    Args:
        query: SQL SELECT query to execute

    Returns:
        JSON string with query results
    """
    try:
        with _safe_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                results = cur.fetchall()
                return json.dumps([dict(row) for row in results], default=str)

    except Exception as e:
        return f"Error executing query: {str(e)}"


@tool
def analyze_watchlist_tool(watchlist_name: str, min_score: float = 60.0) -> str:
    """
    Analyze all stocks in a watchlist for trading opportunities.
    Returns ranked strategies with real option prices.

    Args:
        watchlist_name: Name of watchlist to analyze (e.g., 'NVDA', 'Tech')
        min_score: Minimum profit score threshold (0-100)

    Returns:
        JSON string with top opportunities
    """
    try:
        from src.watchlist_strategy_analyzer import WatchlistStrategyAnalyzer

        analyzer = WatchlistStrategyAnalyzer()
        results = analyzer.analyze_watchlist(
            watchlist_name=watchlist_name,
            min_score=min_score,
            strategies=['CSP', 'CC']
        )

        top_5 = results[:5]
        formatted = []

        for analysis in top_5:
            formatted.append({
                'ticker': analysis.ticker,
                'strategy': analysis.strategy_type,
                'score': round(analysis.profit_score, 1),
                'trade': analysis.trade_details,
                'premium': f"${analysis.expected_premium:.0f}",
                'probability': f"{analysis.probability_profit:.0f}%",
                'recommendation': analysis.recommendation
            })

        return json.dumps(formatted, indent=2)

    except Exception as e:
        return f"Error analyzing watchlist: {str(e)}"


@tool
def get_portfolio_status_tool() -> str:
    """
    Get current portfolio status from Robinhood.
    Shows balance, positions, and P/L.

    Returns:
        JSON string with portfolio information
    """
    try:
        import robin_stocks.robinhood as rh

        username = os.getenv('ROBINHOOD_USERNAME')
        password = os.getenv('ROBINHOOD_PASSWORD')

        if username and password:
            rh.login(username, password)

        account = rh.profiles.load_account_profile()
        positions = rh.get_open_stock_positions()
        options = rh.get_open_option_positions()

        portfolio_data = {
            'balance': account.get('portfolio_cash', '0'),
            'buying_power': account.get('buying_power', '0'),
            'stock_positions': len(positions),
            'option_positions': len(options),
            'equity': account.get('equity', '0')
        }

        return json.dumps(portfolio_data, indent=2)

    except Exception as e:
        return f"Error getting portfolio: {str(e)}"


@tool
def create_task_tool(title: str, description: str, priority: str = "medium") -> str:
    """
    Create a new task in the Magnus task management system.
    Use this when user wants to improve something or add a feature.

    Args:
        title: Short task title
        description: Detailed task description
        priority: Task priority (low, medium, high, critical)

    Returns:
        Success message with task ID
    """
    try:
        from src.task_db_manager import TaskDBManager

        task_mgr = TaskDBManager()
        task_id = task_mgr.create_task(
            title=title,
            description=description,
            task_type='enhancement',
            priority=priority,
            status='pending',
            assigned_agent='auto',
            source='AVA'
        )

        return f"Task #{task_id} created successfully: {title}"

    except Exception as e:
        return f"Error creating task: {str(e)}"


@tool
def get_stock_price_tool(ticker: str) -> str:
    """
    Get current stock price for a ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Current price and details
    """
    try:
        import robin_stocks.robinhood as rh

        quote = rh.get_quotes(ticker)

        if quote and len(quote) > 0:
            price_data = {
                'ticker': ticker,
                'price': quote[0].get('last_trade_price', 'N/A'),
                'previous_close': quote[0].get('previous_close', 'N/A'),
                'change': quote[0].get('change', 'N/A'),
                'change_percentage': quote[0].get('change_percentage', 'N/A')
            }
            return json.dumps(price_data, indent=2)
        else:
            return f"Could not get price for {ticker}"

    except Exception as e:
        return f"Error getting stock price: {str(e)}"


@tool
def search_magnus_knowledge_tool(question: str) -> str:
    """
    Search Magnus project knowledge base for information.
    Use this to answer questions about Magnus features, code, usage.

    Args:
        question: Question about Magnus

    Returns:
        Answer from knowledge base
    """
    try:
        from src.ava.enhanced_project_handler import EnhancedProjectHandler

        handler = EnhancedProjectHandler()
        result = handler.answer_project_question(question)

        if result['success']:
            return result['answer']
        else:
            return f"Could not find answer. Error: {result.get('error', 'Unknown')}"

    except Exception as e:
        return f"Error searching knowledge: {str(e)}"


# ==================== SPORTS BETTING TOOLS ====================

@tool
def get_kalshi_markets_tool(sport: str = "nfl", status: str = "active", limit: int = 10) -> str:
    """
    Get Kalshi prediction markets for NFL or NCAA football.
    
    Args:
        sport: 'nfl' or 'cfb' (college football)
        status: 'active', 'open', 'closed', or 'all'
        limit: Maximum number of markets to return
    
    Returns:
        JSON string with market data including ticker, title, prices, volume
    """
    try:
        from src.kalshi_db_manager import KalshiDBManager
        
        db = KalshiDBManager()
        markets = db.get_active_markets(market_type=sport)
        
        # Apply limit
        if limit:
            markets = markets[:limit]
        
        formatted = []
        for market in markets:
            formatted.append({
                'ticker': market.get('ticker'),
                'title': market.get('title'),
                'yes_price': float(market.get('yes_price', 0)) * 100 if market.get('yes_price') else 0,
                'no_price': float(market.get('no_price', 0)) * 100 if market.get('no_price') else 0,
                'volume': float(market.get('volume', 0)),
                'status': market.get('status'),
                'close_time': str(market.get('close_time', ''))
            })
        
        return json.dumps(formatted, indent=2, default=str)
    
    except Exception as e:
        return f"Error getting Kalshi markets: {str(e)}"


@tool
def get_live_games_tool(sport: str = "nfl") -> str:
    """
    Get live ESPN game scores for NFL or NCAA football.
    
    Args:
        sport: 'nfl' or 'cfb' (college football)
    
    Returns:
        JSON string with live game data including teams, scores, status
    """
    try:
        if sport.lower() == 'cfb':
            from src.espn_ncaa_live_data import get_espn_ncaa_client
            espn = get_espn_ncaa_client()
            games = espn.get_scoreboard(group='80')  # FBS games
        else:
            from src.espn_live_data import get_espn_client
            espn = get_espn_client()
            games = espn.get_scoreboard()
        
        formatted = []
        for game in games:
            formatted.append({
                'game_id': str(game.get('game_id', '')),
                'away_team': game.get('away_team', ''),
                'home_team': game.get('home_team', ''),
                'away_score': game.get('away_score', 0),
                'home_score': game.get('home_score', 0),
                'status': game.get('status_detail', ''),
                'is_live': game.get('is_live', False),
                'clock': game.get('clock', ''),
                'period': game.get('period', 0)
            })
        
        return json.dumps(formatted, indent=2, default=str)
    
    except Exception as e:
        return f"Error getting live games: {str(e)}"


@tool
def get_game_watchlist_tool(user_id: str = "default_user") -> str:
    """
    Get user's watched games with Telegram alerts enabled.
    
    Args:
        user_id: User identifier (defaults to 'default_user')
    
    Returns:
        JSON string with watched games and selected teams
    """
    try:
        from src.game_watchlist_manager import GameWatchlistManager
        
        watchlist_manager = GameWatchlistManager()
        watchlist = watchlist_manager.get_user_watchlist(user_id)
        
        formatted = []
        for game in watchlist:
            formatted.append({
                'game_id': game.get('game_id', ''),
                'away_team': game.get('away_team', ''),
                'home_team': game.get('home_team', ''),
                'selected_team': game.get('selected_team', ''),
                'sport': game.get('sport', ''),
                'added_at': str(game.get('added_at', ''))
            })
        
        return json.dumps(formatted, indent=2, default=str)
    
    except Exception as e:
        return f"Error getting game watchlist: {str(e)}"


@tool
def get_betting_opportunities_tool(min_ev: float = 5.0, sport: str = "nfl") -> str:
    """
    Find high expected value betting opportunities.
    
    Args:
        min_ev: Minimum expected value percentage (default 5.0)
        sport: 'nfl' or 'cfb'
    
    Returns:
        JSON string with betting opportunities ranked by EV
    """
    try:
        from src.espn_live_data import get_espn_client
        from src.espn_ncaa_live_data import get_espn_ncaa_client
        from src.espn_kalshi_matcher_optimized import enrich_games_with_kalshi_odds_optimized as enrich_games_with_kalshi_odds
        from src.advanced_betting_ai_agent import AdvancedBettingAIAgent
        
        # Get games
        if sport.lower() == 'cfb':
            espn = get_espn_ncaa_client()
            games = espn.get_scoreboard(group='80')
        else:
            espn = get_espn_client()
            games = espn.get_scoreboard()
        
        # Enrich with Kalshi odds
        games = enrich_games_with_kalshi_odds(games)
        
        # Get AI predictions
        ai_agent = AdvancedBettingAIAgent()
        opportunities = []
        
        for game in games:
            market_data = game.get('kalshi_odds', {})
            if not market_data:
                continue
            
            prediction = ai_agent.analyze_betting_opportunity(game, market_data)
            ev = prediction.get('expected_value', 0)
            
            if ev >= min_ev:
                opportunities.append({
                    'game': f"{game.get('away_team', '')} @ {game.get('home_team', '')}",
                    'predicted_winner': prediction.get('predicted_winner', ''),
                    'win_probability': round(prediction.get('win_probability', 0) * 100, 1),
                    'confidence': round(prediction.get('confidence_score', 0), 1),
                    'expected_value': round(ev, 1),
                    'recommendation': prediction.get('recommendation', 'PASS'),
                    'kalshi_odds': {
                        'away': round(market_data.get('away_win_price', 0) * 100, 0),
                        'home': round(market_data.get('home_win_price', 0) * 100, 0)
                    }
                })
        
        # Sort by EV descending
        opportunities.sort(key=lambda x: x['expected_value'], reverse=True)
        
        return json.dumps(opportunities[:10], indent=2, default=str)
    
    except Exception as e:
        return f"Error finding betting opportunities: {str(e)}"


# ==================== TRADING TOOLS ====================

@tool
def get_positions_tool() -> str:
    """
    Get current stock and options positions from database.
    
    Returns:
        JSON string with positions including symbol, shares, cost basis
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'magnus'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT symbol, shares, cost_basis, purchase_date, account_type, notes
                FROM stock_holdings
                WHERE active = true
                ORDER BY symbol
            """)
            results = cur.fetchall()
            conn.close()
            
            return json.dumps([dict(row) for row in results], default=str, indent=2)
    
    except Exception as e:
        return f"Error getting positions: {str(e)}"


@tool
def get_trading_opportunities_tool(strategy: str = "CSP", min_score: float = 60.0) -> str:
    """
    Find trading opportunities (CSP, CC, etc.) from database.
    
    Args:
        strategy: 'CSP' (Cash-Secured Puts) or 'CC' (Covered Calls)
        min_score: Minimum profit score (0-100)
    
    Returns:
        JSON string with top opportunities
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'magnus'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT symbol, strategy, strike, premium, expiration, 
                       confidence_score, expected_return, analysis_data
                FROM opportunities
                WHERE strategy = %s 
                  AND confidence_score >= %s
                  AND executed = false
                ORDER BY expected_return DESC
                LIMIT 10
            """, (strategy, int(min_score)))
            results = cur.fetchall()
            conn.close()
            
            return json.dumps([dict(row) for row in results], default=str, indent=2)
    
    except Exception as e:
        return f"Error getting trading opportunities: {str(e)}"


@tool
def get_trade_history_tool(symbol: str = None, limit: int = 10) -> str:
    """
    Get trade history from database.
    
    Args:
        symbol: Optional stock symbol to filter by
        limit: Maximum number of trades to return
    
    Returns:
        JSON string with trade history
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'magnus'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if symbol:
                cur.execute("""
                    SELECT symbol, strategy, strike, premium, quantity, 
                           expiration, entry_date, exit_date, profit_loss, exit_reason
                    FROM trade_history
                    WHERE symbol = %s
                    ORDER BY entry_date DESC
                    LIMIT %s
                """, (symbol.upper(), limit))
            else:
                cur.execute("""
                    SELECT symbol, strategy, strike, premium, quantity, 
                           expiration, entry_date, exit_date, profit_loss, exit_reason
                    FROM trade_history
                    ORDER BY entry_date DESC
                    LIMIT %s
                """, (limit,))
            
            results = cur.fetchall()
            conn.close()
            
            return json.dumps([dict(row) for row in results], default=str, indent=2)
    
    except Exception as e:
        return f"Error getting trade history: {str(e)}"


# ==================== TASK MANAGEMENT TOOLS ====================

@tool
def get_tasks_tool(status: str = None, priority: str = None, limit: int = 10) -> str:
    """
    Get development tasks from database.
    
    Args:
        status: Filter by status ('pending', 'in_progress', 'completed', etc.)
        priority: Filter by priority ('low', 'medium', 'high', 'critical')
        limit: Maximum number of tasks to return
    
    Returns:
        JSON string with tasks
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'magnus'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = "SELECT id, title, description, status, priority, assigned_agent, created_at FROM development_tasks WHERE 1=1"
            params = []
            
            if status:
                query += " AND status = %s"
                params.append(status)
            
            if priority:
                query += " AND priority = %s"
                params.append(priority)
            
            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)
            
            cur.execute(query, params)
            results = cur.fetchall()
            conn.close()
            
            return json.dumps([dict(row) for row in results], default=str, indent=2)
    
    except Exception as e:
        return f"Error getting tasks: {str(e)}"


# ==================== XTRADES TOOLS ====================

@tool
def get_xtrades_profiles_tool(active_only: bool = True) -> str:
    """
    Get monitored Xtrades profiles.
    
    Args:
        active_only: Only return active profiles
    
    Returns:
        JSON string with profile information
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'magnus'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if active_only:
                cur.execute("""
                    SELECT id, username, display_name, active, added_date, 
                           last_sync, total_trades_scraped
                    FROM xtrades_profiles
                    WHERE active = true
                    ORDER BY username
                """)
            else:
                cur.execute("""
                    SELECT id, username, display_name, active, added_date, 
                           last_sync, total_trades_scraped
                    FROM xtrades_profiles
                    ORDER BY username
                """)
            
            results = cur.fetchall()
            conn.close()
            
            return json.dumps([dict(row) for row in results], default=str, indent=2)
    
    except Exception as e:
        return f"Error getting Xtrades profiles: {str(e)}"


@tool
def get_xtrades_trades_tool(profile: str = None, limit: int = 10) -> str:
    """
    Get recent trades from Xtrades profiles.
    
    Args:
        profile: Optional username to filter by
        limit: Maximum number of trades to return
    
    Returns:
        JSON string with recent trades
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'magnus'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if profile:
                cur.execute("""
                    SELECT t.*, p.username, p.display_name
                    FROM xtrades_trades t
                    JOIN xtrades_profiles p ON t.profile_id = p.id
                    WHERE p.username = %s
                    ORDER BY t.scraped_at DESC
                    LIMIT %s
                """, (profile, limit))
            else:
                cur.execute("""
                    SELECT t.*, p.username, p.display_name
                    FROM xtrades_trades t
                    JOIN xtrades_profiles p ON t.profile_id = p.id
                    ORDER BY t.scraped_at DESC
                    LIMIT %s
                """, (limit,))
            
            results = cur.fetchall()
            conn.close()
            
            return json.dumps([dict(row) for row in results], default=str, indent=2)
    
    except Exception as e:
        return f"Error getting Xtrades trades: {str(e)}"

