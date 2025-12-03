"""
Omnipresent AVA - Simplified Version (No LangChain Required)
============================================================

AVA appears at the top of every Magnus page as an expandable assistant.
This version works without LangChain dependencies.

Features:
- Expandable chat interface on all pages
- Direct function calls (no agent framework)
- Database access for queries and task creation
- Watchlist analysis
- Portfolio management
- Memory/recall across sessions

Author: Magnus Trading Platform
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

# Magnus imports
from src.ava.conversation_memory_manager import ConversationMemoryManager
from src.watchlist_strategy_analyzer import WatchlistStrategyAnalyzer
from src.task_db_manager import TaskDBManager
import psycopg2
from psycopg2.extras import RealDictCursor
import json

# Personality system (optional)
try:
    from src.ava.ava_personality import AVAPersonality, PersonalityMode, EmotionalState
    PERSONALITY_AVAILABLE = True
except ImportError:
    PERSONALITY_AVAILABLE = False
    AVAPersonality = PersonalityMode = EmotionalState = None

# RAG system (optional)
try:
    from src.rag.unified_rag import get_unified_rag
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Autonomous Task System (optional)
try:
    from src.ava.autonomous_task_system import get_task_system
    TASK_SYSTEM_AVAILABLE = True
except ImportError:
    TASK_SYSTEM_AVAILABLE = False
    get_task_system = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleAVA:
    """Simplified AVA without LangChain dependencies"""

    def __init__(self) -> None:
        """Initialize simple AVA with personality and RAG support"""
        self.memory_manager = ConversationMemoryManager()

        # Initialize personality system
        if PERSONALITY_AVAILABLE:
            self.personality = AVAPersonality(mode=PersonalityMode.FRIENDLY)
            logger.info("✅ Personality system initialized")
        else:
            self.personality = None
            logger.info("ℹ️ Personality system not available")

        # Initialize RAG system
        if RAG_AVAILABLE:
            try:
                self.rag = get_unified_rag()
                logger.info("✅ RAG system initialized")
            except Exception as e:
                self.rag = None
                logger.warning(f"⚠️ RAG initialization failed: {e}")
        else:
            self.rag = None
            logger.info("ℹ️ RAG system not available")

        # Initialize Autonomous Task System
        if TASK_SYSTEM_AVAILABLE and get_task_system:
            try:
                self.task_system = get_task_system(auto_execute=False)
                logger.info("✅ Autonomous Task System initialized")
            except Exception as e:
                self.task_system = None
                logger.warning(f"⚠️ Task system initialization failed: {e}")
        else:
            self.task_system = None
            logger.info("ℹ️ Autonomous Task System not available")

    def set_personality(self, mode_name: str) -> bool:
        """Change personality mode"""
        if not PERSONALITY_AVAILABLE or not self.personality:
            return False
        try:
            mode = PersonalityMode(mode_name.lower())
            self.personality.set_mode(mode)
            return True
        except (ValueError, KeyError):
            return False

    def apply_personality(self, response: str, context: dict = None) -> str:
        """Apply personality styling to response"""
        if self.personality:
            try:
                if context:
                    emotional_state = self.personality.detect_emotional_context(context)
                    self.personality.set_emotional_state(emotional_state)
                return self.personality.style_response(response, context or {})
            except Exception as e:
                logger.warning(f"Personality styling failed: {e}")
        return response

    def query_rag(self, question: str) -> Optional[str]:
        """Query RAG knowledge base"""
        if not self.rag:
            return None
        try:
            result = self.rag.query(question)
            if result and result.confidence >= 0.5:
                return result.answer
        except Exception as e:
            logger.warning(f"RAG query failed: {e}")
        return None

    def query_database(self, query: str) -> str:
        """Execute SQL query on Magnus database"""
        try:
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                database=os.getenv('DB_NAME', 'magnus'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', '')
            )
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query)
            results = cur.fetchall()
            conn.close()
            return json.dumps([dict(row) for row in results], default=str)
        except Exception as e:
            return f"Database error: {str(e)}"

    def create_task(self, title: str, description: str, priority: str = "medium") -> str:
        """Create new task in Magnus"""
        try:
            task_mgr = TaskDBManager()
            task_id = task_mgr.create_task(
                title=title,
                description=description,
                priority=priority,
                assigned_agent='user_requested'
            )
            return f"✓ Created task #{task_id}: {title}"
        except Exception as e:
            return f"Error creating task: {str(e)}"

    def analyze_watchlist(self, watchlist_name: str) -> str:
        """Analyze watchlist for opportunities"""
        try:
            analyzer = WatchlistStrategyAnalyzer()
            results = analyzer.analyze_watchlist(watchlist_name, min_score=60.0)
            if results:
                summary = f"Found {len(results)} opportunities in {watchlist_name}:\n\n"
                for r in results[:5]:  # Top 5
                    summary += f"• {r['ticker']}: Score {r['score']:.0f}/100\n"
                return summary
            return f"No opportunities found in {watchlist_name}"
        except Exception as e:
            return f"Error analyzing watchlist: {str(e)}"

    def get_portfolio_status(self) -> str:
        """Get Robinhood portfolio status"""
        try:
            import robin_stocks.robinhood as rh
            username = os.getenv('ROBINHOOD_USERNAME')
            password = os.getenv('ROBINHOOD_PASSWORD')

            if not username or not password:
                return "Robinhood credentials not configured"

            rh.login(username, password)
            account = rh.profiles.load_account_profile()

            portfolio_value = float(account.get('portfolio_cash', 0))
            return f"Portfolio Value: ${portfolio_value:,.2f}"
        except Exception as e:
            return f"Error getting portfolio: {str(e)}"

    def get_stock_price(self, ticker: str) -> str:
        """Get current stock price"""
        try:
            import robin_stocks.robinhood as rh
            quote = rh.get_quotes(ticker)
            if quote and len(quote) > 0:
                price = float(quote[0].get('last_trade_price', 0))
                return f"{ticker}: ${price:.2f}"
            return f"Unable to get price for {ticker}"
        except Exception as e:
            return f"Error getting stock price: {str(e)}"

    def process_message(self, user_message: str, user_id: str = "web_user", platform: str = "web") -> Dict:
        """Process user message and return response"""

        # Start or get conversation
        conversation_id = self.memory_manager.get_active_conversation(user_id, platform)
        if not conversation_id:
            conversation_id = self.memory_manager.start_conversation(user_id, platform)

        # Simple intent detection
        message_lower = user_message.lower()
        response = None
        intent = "unknown"
        action = None
        success = True

        try:
            # AUTONOMOUS TASK SYSTEM - Check for "task:" commands first
            if self.task_system and (message_lower.startswith('task:') or message_lower.startswith('task ')):
                intent = "autonomous_task"
                task_result = self.task_system.process_message(user_message, user_id)

                if task_result.get('is_task'):
                    if task_result.get('success'):
                        response = f"""**Autonomous Task Created**

**Task #{task_result.get('task_id')}** - {task_result.get('task_type', 'general').replace('_', ' ').title()}

{task_result.get('message', '')}"""

                        if task_result.get('files_modified'):
                            response += f"\n\n**Files Modified:** {', '.join(task_result['files_modified'])}"

                        if task_result.get('execution_time'):
                            response += f"\n**Execution Time:** {task_result['execution_time']:.1f}s"

                        action = "autonomous_task_executed"
                    else:
                        response = f"Task creation failed: {task_result.get('message', 'Unknown error')}"
                        success = False
                else:
                    response = "Could not parse task. Try: 'task: add a new feature to...'"
                    success = False

            # Database queries
            elif any(word in message_lower for word in ['query', 'database', 'select', 'show me', 'how many']):
                intent = "database_query"
                if 'select' in message_lower:
                    # Extract SQL query
                    query_start = message_lower.find('select')
                    query = user_message[query_start:]
                    response = self.query_database(query)
                    action = "database_query_executed"
                else:
                    response = "To query the database, please provide a SELECT statement."

            # Task creation
            elif any(word in message_lower for word in ['create task', 'add task', 'new task', 'improve']):
                intent = "create_task"
                # Extract task title from message
                task_title = user_message.replace('create task', '').replace('add task', '').strip()
                if not task_title:
                    task_title = "User requested improvement"
                response = self.create_task(
                    title=task_title,
                    description=f"User request: {user_message}",
                    priority="medium"
                )
                action = "task_created"

            # Watchlist analysis
            elif any(word in message_lower for word in ['analyze', 'watchlist', 'opportunities']):
                intent = "analyze_watchlist"
                # Try to extract watchlist name
                for word in ['nvda', 'aapl', 'tech', 'wheel']:
                    if word in message_lower:
                        response = self.analyze_watchlist(word.upper())
                        action = "watchlist_analyzed"
                        break
                if not response:
                    response = "Please specify a watchlist name (e.g., 'analyze NVDA watchlist')"

            # Portfolio status
            elif any(word in message_lower for word in ['portfolio', 'balance', 'account']):
                intent = "portfolio_status"
                response = self.get_portfolio_status()
                action = "portfolio_checked"

            # Stock price
            elif any(word in message_lower for word in ['price of', 'stock price', 'what is', "what's"]):
                intent = "stock_price"
                # Extract ticker
                words = user_message.upper().split()
                for word in words:
                    if len(word) <= 5 and word.isalpha():
                        response = self.get_stock_price(word)
                        action = "stock_price_fetched"
                        break
                if not response:
                    response = "Please specify a ticker symbol (e.g., 'price of AAPL')"

            # General help
            elif any(word in message_lower for word in ['help', 'what can you', 'how do i']):
                intent = "help"
                response = """I can help you with:

**Database Queries:**
- "Show me all pending tasks"
- "Query: SELECT * FROM watchlists"

**Task Management:**
- "Create task to improve dashboard"
- "Add task for better analysis"

**Watchlist Analysis:**
- "Analyze NVDA watchlist"
- "Show opportunities in TECH"

**Portfolio:**
- "What's my portfolio balance?"
- "Check my account"

**Stock Prices:**
- "What's the price of AAPL?"
- "Stock price TSLA"

**Magnus Information:**
- "What is Magnus?"
- "How does Magnus work?"
"""
                action = "help_provided"

            # About Magnus
            elif any(word in message_lower for word in ['what is magnus', 'about magnus', 'tell me about']):
                intent = "about_magnus"
                response = """**Magnus** is an advanced options trading platform focused on the Wheel Strategy.

**Key Features:**
- 📊 Real-time position tracking from Robinhood
- 📈 TradingView watchlist integration
- 💰 Premium collection tracking
- 📉 Theta decay forecasting
- 🎯 AI-powered trade recommendations
- 🔍 Database scanning for opportunities
- 📅 Earnings calendar tracking
- 🎲 Prediction markets integration

I'm AVA, your AI assistant. I can help you analyze positions, create tasks, query data, and more!
"""
                action = "about_provided"

            # Default response - try RAG first, then provide helpful suggestions
            else:
                intent = "general_conversation"
                query_lower = user_message.lower()

                # First, try RAG knowledge base
                rag_answer = self.query_rag(user_message)
                if rag_answer:
                    response = rag_answer
                    action = "rag_response"
                    success = True
                # Provide context-aware suggestions
                elif any(w in query_lower for w in ['option', 'put', 'call', 'wheel', 'spread']):
                    response = """I can help with options! Try asking:
• "What is the wheel strategy?"
• "Explain covered calls"
• "Find CSP opportunities"
• "What are the Greeks?"

Or ask about a specific stock's options!"""
                    action = "options_suggestions"
                    success = True  # Helpful suggestions count as success
                elif any(w in query_lower for w in ['portfolio', 'position', 'balance']):
                    response = """I can help with your portfolio! Try:
• "Show my portfolio"
• "What positions do I have?"
• "Show my balance"
• "Analyze my positions" """
                    action = "portfolio_suggestions"
                    success = True
                elif any(w in query_lower for w in ['stock', 'price', 'chart']):
                    response = """I can help with stocks! Try:
• "What's AAPL at?"
• "Show NVDA chart"
• "Analyze TSLA" """
                    action = "stock_suggestions"
                    success = True
                elif any(w in query_lower for w in ['personality', 'mode', 'style']):
                    # Handle personality change requests
                    personalities = ['professional', 'friendly', 'witty', 'mentor', 'concise',
                                   'charming', 'analyst', 'coach', 'rebel', 'guru']
                    for p in personalities:
                        if p in query_lower:
                            if self.set_personality(p):
                                response = f"✅ Personality changed to **{p.title()}**! {self.personality.get_greeting() if self.personality else ''}"
                                action = "personality_changed"
                                success = True
                                break
                    else:
                        response = f"""I can change my personality style! Available modes:

• **professional** - Formal & data-focused
• **friendly** - Warm & approachable (default)
• **witty** - Clever & humorous
• **mentor** - Educational & patient
• **analyst** - Bloomberg style
• **coach** - Motivational
• **rebel** - Contrarian
• **guru** - Zen master

Say "Change to coach mode" or "Be more witty" to switch!"""
                        action = "personality_help"
                        success = True
                else:
                    response = """I'm here to help! Here's what I can do:

**Quick Commands:**
• "Show my portfolio" - See your positions
• "Find opportunities" - Scan for trades
• "What's AAPL at?" - Get stock prices
• "Explain wheel strategy" - Learn options

Type "help" for all commands!"""
                    action = "general_help"
                    success = False  # Only mark as failed if completely unhelpful

                    # Log as unanswered question only if we couldn't help at all
                    self.memory_manager.record_unanswered_question(
                        user_question=user_message,
                        intent_detected=intent,
                        confidence_score=0.0,
                        failure_reason="unsupported_query",
                        conversation_id=conversation_id,
                        context={'platform': platform}
                    )

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            response = f"I encountered an error: {str(e)}"
            success = False

        # Apply personality styling to response
        if response:
            response = self.apply_personality(response)

        # Log the interaction
        self.memory_manager.log_message(
            conversation_id=conversation_id,
            user_message=user_message,
            ava_response=response,
            intent_detected=intent,
            confidence_score=0.8 if success else 0.2,
            action_performed=action,
            action_success=success,
            model_used='simple_ava'
        )

        return {
            'response': response,
            'success': success,
            'intent': intent,
            'conversation_id': conversation_id
        }


def show_omnipresent_ava():
    """
    Display AVA as an expandable assistant at the top of every page.
    This is the main UI component to add to all Magnus pages.
    """

    # Custom CSS for AVA
    st.markdown("""
    <style>
    .ava-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .ava-message {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .user-message {
        background-color: #e3f2fd;
        text-align: right;
    }
    .ava-response {
        background-color: #f5f5f5;
        text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize AVA in session state
    if 'simple_ava' not in st.session_state:
        st.session_state.simple_ava = SimpleAVA()

    # Initialize message history
    if 'ava_messages' not in st.session_state:
        st.session_state.ava_messages = []

    # Expandable AVA interface
    with st.expander("🤖 **AVA - Your Expert Trading Assistant**", expanded=False):
        st.caption("Ask me anything about Magnus, your portfolio, watchlists, or create tasks!")

        # Display chat history
        for msg in st.session_state.ava_messages[-10:]:  # Last 10 messages
            if msg['role'] == 'user':
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**AVA:** {msg['content']}")

        # Chat input
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input(
                "Ask AVA:",
                key="ava_input",
                placeholder="e.g., 'analyze NVDA watchlist' or 'create task to improve dashboard'"
            )
        with col2:
            send_button = st.button("Send", type="primary")

        if send_button and user_input:
            # Add user message
            st.session_state.ava_messages.append({
                'role': 'user',
                'content': user_input
            })

            # Get AVA response
            ava = st.session_state.simple_ava
            response_data = ava.process_message(
                user_input,
                user_id=st.session_state.get('user_id', 'web_user'),
                platform='web'
            )

            # Add AVA response
            st.session_state.ava_messages.append({
                'role': 'ava',
                'content': response_data['response']
            })

            # Rerun to show new messages
            st.rerun()

        # Quick action buttons
        st.markdown("---")
        st.caption("**Quick Actions:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("💼 Portfolio"):
                st.session_state.ava_messages.append({'role': 'user', 'content': 'Check my portfolio'})
                ava = st.session_state.simple_ava
                response = ava.process_message('Check my portfolio', 'web_user', 'web')
                st.session_state.ava_messages.append({'role': 'ava', 'content': response['response']})
                st.rerun()
        with col2:
            if st.button("📊 Help"):
                st.session_state.ava_messages.append({'role': 'user', 'content': 'help'})
                ava = st.session_state.simple_ava
                response = ava.process_message('help', 'web_user', 'web')
                st.session_state.ava_messages.append({'role': 'ava', 'content': response['response']})
                st.rerun()
        with col3:
            if st.button("🔍 About"):
                st.session_state.ava_messages.append({'role': 'user', 'content': 'What is Magnus?'})
                ava = st.session_state.simple_ava
                response = ava.process_message('What is Magnus?', 'web_user', 'web')
                st.session_state.ava_messages.append({'role': 'ava', 'content': response['response']})
                st.rerun()
