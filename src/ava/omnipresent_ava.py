"""
Omnipresent AVA - Expert AI Assistant with Full System Access
=============================================================

AVA appears at the top of every page as an expandable assistant.
Has full access to all platform features via LangChain tools.

Features:
- Expandable chat interface on all pages
- LangChain agent with custom tools
- Database access for queries and task creation
- Watchlist analysis
- Portfolio management
- Complete platform knowledge
- Memory/recall across sessions

Author: AVA Trading Platform
Created: 2025-11-11
"""

import streamlit as st
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# LangChain imports
try:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_core.tools import tool
    from langchain_community.llms import Ollama
    from langchain_groq import ChatGroq
    from langchain.prompts import PromptTemplate
    from langchain.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        # Fallback: try langchain.tools
        from langchain.agents import AgentExecutor, create_react_agent
        from langchain.tools import tool
        from langchain_community.llms import Ollama
        from langchain_groq import ChatGroq
        from langchain.prompts import PromptTemplate
        from langchain.memory import ConversationBufferMemory
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        # Create a dummy decorator to prevent NameError
        def tool(func):
            return func
        logging.warning("LangChain not available. Install: pip install langchain langchain-community langchain-groq")

# Magnus imports
from src.ava.conversation_memory_manager import ConversationMemoryManager
from src.watchlist_strategy_analyzer import WatchlistStrategyAnalyzer
from src.task_db_manager import TaskDBManager
from src.services.llm_service import LLMService
import psycopg2
from psycopg2.extras import RealDictCursor
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# LANGCHAIN TOOLS - AVA's Capabilities
# ============================================================================

@tool
def query_database(query: str) -> str:
    """
    Execute a SQL query on the Magnus database.
    Use this to get information about tasks, positions, watchlists, etc.

    Args:
        query: SQL SELECT query to execute

    Returns:
        JSON string with query results
    """
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        import json
        import os

        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME', 'magnus'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', '')
        )

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            results = cur.fetchall()
            conn.close()

            # Convert to JSON-serializable format
            return json.dumps([dict(row) for row in results], default=str)

    except Exception as e:
        return f"Error executing query: {str(e)}"


@tool
def create_task(title: str, description: str, priority: str = "medium") -> str:
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
def analyze_watchlist(watchlist_name: str, min_score: float = 60.0) -> str:
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
        import json

        analyzer = WatchlistStrategyAnalyzer()
        results = analyzer.analyze_watchlist(
            watchlist_name=watchlist_name,
            min_score=min_score,
            strategies=['CSP', 'CC']
        )

        # Format top 5 results
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
def get_portfolio_status() -> str:
    """
    Get current portfolio status from Robinhood.
    Shows balance, positions, and P/L.

    Returns:
        JSON string with portfolio information
    """
    try:
        import robin_stocks.robinhood as rh
        import json
        import os

        # Login if needed
        username = os.getenv('ROBINHOOD_USERNAME')
        password = os.getenv('ROBINHOOD_PASSWORD')

        if username and password:
            rh.login(username, password)

        # Get account data
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
def search_magnus_knowledge(question: str) -> str:
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


@tool
def get_recent_tasks(limit: int = 10, status: str = None) -> str:
    """
    Get recent tasks from Magnus task management system.

    Args:
        limit: Number of tasks to return
        status: Filter by status (pending, in_progress, completed, etc.)

    Returns:
        JSON string with task list
    """
    try:
        from src.task_db_manager import TaskDBManager
        import json

        task_mgr = TaskDBManager()
        tasks = task_mgr.get_recent_tasks(limit=limit, status=status)

        formatted = []
        for task in tasks:
            formatted.append({
                'id': task['task_id'],
                'title': task['title'],
                'status': task['status'],
                'priority': task['priority'],
                'created': str(task['created_at'])
            })

        return json.dumps(formatted, indent=2)

    except Exception as e:
        return f"Error getting tasks: {str(e)}"


@tool
def update_task_status(task_id: int, new_status: str, notes: str = None) -> str:
    """
    Update the status of an existing task.

    Args:
        task_id: Task ID to update
        new_status: New status (pending, in_progress, completed, cancelled)
        notes: Optional notes about the update

    Returns:
        Success message
    """
    try:
        from src.task_db_manager import TaskDBManager

        task_mgr = TaskDBManager()
        task_mgr.update_task(task_id, status=new_status, notes=notes)

        return f"Task #{task_id} updated to status: {new_status}"

    except Exception as e:
        return f"Error updating task: {str(e)}"


@tool
def get_stock_price(ticker: str) -> str:
    """
    Get current stock price for a ticker.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Current price and details
    """
    try:
        import robin_stocks.robinhood as rh
        import json

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


# ============================================================================
# OMNIPRESENT AVA CLASS
# ============================================================================

class OmnipresentAVA:
    """AVA assistant that appears on every Magnus page"""

    def __init__(self) -> None:
        """Initialize AVA with LangChain agent"""
        self.memory_manager = ConversationMemoryManager()

        # Initialize LangChain agent if available
        if LANGCHAIN_AVAILABLE:
            self._initialize_langchain_agent()
        else:
            logger.warning("LangChain not available, using fallback mode")
            self.agent = None

        # Fallback LLM service
        self.llm_service = LLMService()

    def _initialize_langchain_agent(self) -> None:
        """Initialize LangChain agent with tools"""
        try:
            # Tools available to AVA
            tools = [
                query_database,
                create_task,
                analyze_watchlist,
                get_portfolio_status,
                search_magnus_knowledge,
                get_recent_tasks,
                update_task_status,
                get_stock_price
            ]

            # Use Groq for fast inference (free tier)
            groq_api_key = os.getenv('GROQ_API_KEY')

            if groq_api_key:
                llm = ChatGroq(
                    model="llama-3.1-70b-versatile",
                    temperature=0.3,
                    groq_api_key=groq_api_key
                )
            else:
                # Fallback to Ollama if available
                llm = Ollama(model="llama3", temperature=0.3)

            # Create custom prompt for AVA
            prompt = PromptTemplate.from_template("""
You are AVA (Automated Vector Agent), the expert AI assistant for Magnus Trading Dashboard.

You have access to these tools:
{tools}

Tool names: {tool_names}

You can:
- Query the database for any information
- Create tasks when users want improvements
- Analyze watchlists for trading opportunities
- Check portfolio status
- Search Magnus knowledge base
- Manage tasks (view, update, create)
- Get stock prices
- Access all Magnus features

When answering:
1. Think step-by-step about what information you need
2. Use tools to get accurate, real-time data
3. Provide specific, actionable answers
4. Reference actual data (task IDs, prices, etc.)
5. Create tasks when users want something improved

Current conversation:
{chat_history}

User: {input}

Thought: {agent_scratchpad}
""")

            # Create agent
            agent = create_react_agent(llm, tools, prompt)

            # Create memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )

            # Create executor
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                memory=memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )

            logger.info("[OK] LangChain agent initialized with tools")

        except Exception as e:
            logger.error(f"Error initializing LangChain agent: {e}")
            self.agent_executor = None

    def process_message(
        self,
        user_message: str,
        user_id: str = 'web_user',
        platform: str = 'web'
    ) -> Dict:
        """
        Process user message with full system access

        Args:
            user_message: User's input
            user_id: User identifier
            platform: Platform (web, telegram, api)

        Returns:
            Response dict with answer and metadata
        """
        # Get or create conversation
        conversation_id = self.memory_manager.get_active_conversation(user_id, platform)

        start_time = datetime.now()

        try:
            # Use LangChain agent if available
            if self.agent_executor:
                response = self.agent_executor.invoke({
                    "input": user_message
                })

                answer = response.get('output', 'I encountered an issue processing your request.')

                # Log successful interaction
                self.memory_manager.log_message(
                    conversation_id=conversation_id,
                    user_message=user_message,
                    ava_response=answer,
                    intent_detected='LANGCHAIN_AGENT',
                    confidence_score=0.95,
                    action_performed='agent_execution',
                    action_success=True,
                    action_duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                    model_used='langchain_agent'
                )

                return {
                    'response': answer,
                    'success': True,
                    'method': 'langchain_agent',
                    'conversation_id': conversation_id
                }

            else:
                # Fallback to basic LLM
                response = self.llm_service.generate(
                    prompt=f"You are AVA, Magnus trading assistant. Answer: {user_message}",
                    max_tokens=500,
                    temperature=0.7
                )

                answer = response['text']

                # Log interaction
                self.memory_manager.log_message(
                    conversation_id=conversation_id,
                    user_message=user_message,
                    ava_response=answer,
                    intent_detected='FALLBACK_LLM',
                    confidence_score=0.7,
                    action_performed='llm_generation',
                    action_success=True,
                    action_duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                    model_used=response.get('model', 'unknown')
                )

                return {
                    'response': answer,
                    'success': True,
                    'method': 'fallback_llm',
                    'conversation_id': conversation_id
                }

        except Exception as e:
            logger.error(f"Error processing message: {e}")

            # Log as unanswered question
            self.memory_manager.record_unanswered_question(
                user_question=user_message,
                intent_detected='ERROR',
                confidence_score=0.0,
                failure_reason='error',
                error_message=str(e),
                user_id=user_id,
                platform=platform,
                conversation_id=conversation_id
            )

            return {
                'response': f"I encountered an error: {str(e)}. I've logged this for improvement.",
                'success': False,
                'error': str(e),
                'conversation_id': conversation_id
            }


# ============================================================================
# STREAMLIT UI COMPONENT
# ============================================================================

def show_omnipresent_ava():
    """
    Render AVA as an expandable component at the top of the page.
    Call this at the start of every Magnus page.
    """
    # Initialize AVA in session state
    if 'omnipresent_ava' not in st.session_state:
        st.session_state.omnipresent_ava = OmnipresentAVA()

    if 'ava_messages' not in st.session_state:
        st.session_state.ava_messages = []

    # Custom CSS for AVA
    st.markdown("""
    <style>
    .ava-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .ava-header {
        color: white;
        font-weight: bold;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

    # AVA Header
    st.markdown('<div class="ava-container">', unsafe_allow_html=True)

    with st.expander("🤖 **AVA - Your Expert Trading Assistant** (Click to chat)", expanded=False):
        st.caption("I have full access to Magnus. Ask me anything or request improvements!")

        # Display chat history
        for msg in st.session_state.ava_messages:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])

        # Chat input
        user_input = st.chat_input("Ask AVA anything...", key="ava_omnipresent_input")

        if user_input:
            # Add user message
            st.session_state.ava_messages.append({
                'role': 'user',
                'content': user_input
            })

            # Display user message
            with st.chat_message('user'):
                st.markdown(user_input)

            # Get AVA response
            with st.chat_message('assistant'):
                with st.spinner("AVA is thinking..."):
                    ava = st.session_state.omnipresent_ava

                    # Get user ID from session
                    user_id = st.session_state.get('user_id', 'web_user_default')

                    response_data = ava.process_message(
                        user_message=user_input,
                        user_id=user_id,
                        platform='web'
                    )

                    response_text = response_data['response']
                    st.markdown(response_text)

                    # Show method used
                    if response_data.get('method'):
                        st.caption(f"Method: {response_data['method']}")

            # Add AVA response to history
            st.session_state.ava_messages.append({
                'role': 'assistant',
                'content': response_text
            })

            # Rerun to update chat
            st.rerun()

        # Quick actions
        st.markdown("---")
        st.caption("**Quick Actions:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Analyze Watchlist", key="ava_quick_analyze"):
                st.session_state.ava_messages.append({
                    'role': 'user',
                    'content': 'Analyze my default watchlist'
                })
                st.rerun()

        with col2:
            if st.button("💼 Check Portfolio", key="ava_quick_portfolio"):
                st.session_state.ava_messages.append({
                    'role': 'user',
                    'content': 'Show my portfolio status'
                })
                st.rerun()

        with col3:
            if st.button("📝 View Tasks", key="ava_quick_tasks"):
                st.session_state.ava_messages.append({
                    'role': 'user',
                    'content': 'Show me recent tasks'
                })
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# Example usage
if __name__ == "__main__":
    # Test AVA locally
    ava = OmnipresentAVA()

    test_queries = [
        "What tasks are pending?",
        "Analyze the NVDA watchlist",
        "Create a task to improve the dashboard UI",
        "What's the current price of TSLA?",
        "Show me my portfolio"
    ]

    for query in test_queries:
        print(f"\n[User] {query}")
        response = ava.process_message(query)
        print(f"[AVA] {response['response']}")
        print(f"[Method] {response.get('method', 'unknown')}")
