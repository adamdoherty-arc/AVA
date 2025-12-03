"""
Omnipresent AVA - Enhanced with Intelligent Question-Asking
============================================================

AVA with improved clarification questions and multi-turn conversations.

Features:
- Context-aware follow-up questions
- Multi-turn conversation tracking
- Smart suggestions based on available data
- Confirmation prompts for critical actions
- Parameter extraction and validation
- Conversation state management

Author: AVA Trading Platform
Created: 2025-11-11
Updated: 2025-11-12 - New AVA avatar integrated
"""

import streamlit as st
import streamlit.components.v1 as components
# Using Streamlit's native st.chat_input (open-source, official component)
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import os
import sys
from pathlib import Path
import re
import time  # PHASE 2: Response time tracking

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Setup logger
logger = logging.getLogger(__name__)

# AVA imports - Make them optional for graceful degradation
try:
    from src.ava.conversation_memory_manager import ConversationMemoryManager
except ImportError:
    ConversationMemoryManager = None

try:
    from src.ava.ava_visual import AvaVisual, AvaExpression, AvaAvatarWidget
except ImportError:
    AvaVisual = AvaExpression = AvaAvatarWidget = None

try:
    from src.ava.ava_personality import AVAPersonality, PersonalityMode, EmotionalState
except ImportError:
    AVAPersonality = PersonalityMode = EmotionalState = None

try:
    from src.ava.web_voice_handler import WebVoiceHandler
except ImportError:
    WebVoiceHandler = None

try:
    from src.watchlist_strategy_analyzer import WatchlistStrategyAnalyzer
except ImportError:
    WatchlistStrategyAnalyzer = None

try:
    from src.task_db_manager import TaskDBManager
except ImportError:
    TaskDBManager = None

# Autonomous Task System Integration
try:
    from src.ava.autonomous_task_system import AutonomousTaskSystem, get_task_system
    TASK_SYSTEM_AVAILABLE = True
except ImportError:
    TASK_SYSTEM_AVAILABLE = False
    AutonomousTaskSystem = None
    get_task_system = None

# PHASE 1 CRITICAL IMPORTS - RAG + LLM Integration (Optional)
try:
    from src.rag.rag_service import RAGService
except ImportError:
    logger.warning("RAGService not available - sentence_transformers not installed")
    RAGService = None

try:
    from src.services.llm_service import LLMService
except ImportError:
    logger.warning("LLMService not available")
    LLMService = None

try:
    from src.magnus_local_llm import MagnusLocalLLM, TaskComplexity
except ImportError:
    logger.warning("MagnusLocalLLM not available")
    MagnusLocalLLM = None
    TaskComplexity = None

import psycopg2
from psycopg2.extras import RealDictCursor
import json
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """States for multi-turn conversations"""
    IDLE = "idle"
    AWAITING_WATCHLIST_NAME = "awaiting_watchlist_name"
    AWAITING_TICKER_SYMBOL = "awaiting_ticker_symbol"
    AWAITING_TASK_DETAILS = "awaiting_task_details"
    AWAITING_TASK_PRIORITY = "awaiting_task_priority"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    AWAITING_SQL_QUERY = "awaiting_sql_query"


class EnhancedAVA:
    """Enhanced AVA with intelligent question-asking"""

    def __init__(self) -> None:
        """Initialize enhanced AVA with RAG and LLM"""
        self.memory_manager = ConversationMemoryManager()

        # Initialize Personality System
        if AVAPersonality is not None and PersonalityMode is not None:
            try:
                default_mode = PersonalityMode.FRIENDLY
                self.personality = AVAPersonality(mode=default_mode)
                logger.info(f"✅ Personality system initialized ({default_mode.value})")
            except Exception as e:
                logger.warning(f"⚠️ Personality init failed: {e}")
                self.personality = None
        else:
            self.personality = None
            logger.info("ℹ️ Personality system not available")

        # PHASE 1: Initialize RAG System (534-line production implementation)
        if RAGService is not None:
            try:
                self.rag_service = RAGService(collection_name='magnus_knowledge')
                logger.info("✅ RAGService initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ RAGService init failed: {e} - Using fallback")
                self.rag_service = None
        else:
            self.rag_service = None
            logger.info("ℹ️ RAGService not available (optional dependency)")

        # PHASE 1: Initialize LLM Service (FREE Groq/Llama-3.3-70b)
        if LLMService is not None:
            try:
                self.llm_service = LLMService()
                logger.info("✅ LLMService initialized with FREE providers")
            except Exception as e:
                logger.warning(f"⚠️ LLMService init failed: {e} - Using fallback")
                self.llm_service = None
        else:
            self.llm_service = None
            logger.info("ℹ️ LLMService not available (optional dependency)")

        # Initialize Local LLM Service (Qwen models via Ollama)
        if MagnusLocalLLM is not None:
            try:
                self.local_llm = MagnusLocalLLM(
                    ollama_host="http://localhost:11434",
                    default_complexity=TaskComplexity.FAST if TaskComplexity else None
                )
                logger.info("✅ Local LLM (Qwen) initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ Local LLM init failed: {e} - Using fallback")
                self.local_llm = None
        else:
            self.local_llm = None
            logger.info("ℹ️ Local LLM not available (optional)")

        # Initialize Autonomous Task System
        if TASK_SYSTEM_AVAILABLE and get_task_system is not None:
            try:
                self.task_system = get_task_system(auto_execute=False)  # Queue tasks for review by default
                logger.info("✅ Autonomous Task System initialized")
            except Exception as e:
                logger.warning(f"⚠️ Task system init failed: {e}")
                self.task_system = None
        else:
            self.task_system = None
            logger.info("ℹ️ Autonomous Task System not available")

    def _log_feedback(self, message_index: int, feedback_type: str):
        """PHASE 2: Log user feedback for continuous improvement"""
        try:
            # Store feedback in database for analytics
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                database=os.getenv('DB_NAME', 'magnus'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', '')
            )
            cur = conn.cursor()

            # Create feedback table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ava_feedback (
                    id SERIAL PRIMARY KEY,
                    message_index INTEGER,
                    feedback_type VARCHAR(20),
                    timestamp TIMESTAMP DEFAULT NOW(),
                    user_message TEXT,
                    ava_response TEXT
                )
            """)

            # Get the message details
            if message_index < len(st.session_state.ava_messages):
                msg = st.session_state.ava_messages[message_index]
                prev_msg = st.session_state.ava_messages[message_index - 1] if message_index > 0 else None

                user_msg = prev_msg.get('content', '') if prev_msg and prev_msg['role'] == 'user' else ''
                ava_msg = msg.get('content', '') if msg['role'] == 'ava' else ''

                # Insert feedback
                cur.execute("""
                    INSERT INTO ava_feedback (message_index, feedback_type, user_message, ava_response)
                    VALUES (%s, %s, %s, %s)
                """, (message_index, feedback_type, user_msg, ava_msg))

                conn.commit()
                logger.info(f"✅ Logged {feedback_type} feedback for message {message_index}")

            conn.close()
        except Exception as e:
            logger.error(f"Error logging feedback: {e}")

    def get_user_preferences(self, user_id: str = "web_user") -> Dict:
        """PHASE 3: Get user preferences from database"""
        try:
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                database=os.getenv('DB_NAME', 'magnus'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', '')
            )
            cur = conn.cursor()

            # Create preferences table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ava_user_preferences (
                    user_id VARCHAR(100) PRIMARY KEY,
                    risk_tolerance VARCHAR(20) DEFAULT 'moderate',
                    favorite_tickers TEXT[],
                    max_position_size INTEGER DEFAULT 5000,
                    preferred_strategy VARCHAR(50) DEFAULT 'wheel',
                    preferences_json JSONB,
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)

            # Get preferences
            cur.execute("""
                SELECT risk_tolerance, favorite_tickers, max_position_size,
                       preferred_strategy, preferences_json
                FROM ava_user_preferences
                WHERE user_id = %s
            """, (user_id,))

            row = cur.fetchone()
            conn.close()

            if row:
                return {
                    'risk_tolerance': row[0],
                    'favorite_tickers': row[1] if row[1] else [],
                    'max_position_size': row[2],
                    'preferred_strategy': row[3],
                    'custom': json.loads(row[4]) if row[4] else {}
                }
            else:
                # Return defaults
                return {
                    'risk_tolerance': 'moderate',
                    'favorite_tickers': [],
                    'max_position_size': 5000,
                    'preferred_strategy': 'wheel',
                    'custom': {}
                }
        except Exception as e:
            logger.error(f"Error getting preferences: {e}")
            return {
                'risk_tolerance': 'moderate',
                'favorite_tickers': [],
                'max_position_size': 5000,
                'preferred_strategy': 'wheel',
                'custom': {}
            }

    def set_user_preference(self, user_id: str, preference_key: str, preference_value: Any):
        """PHASE 3: Set user preference"""
        try:
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                database=os.getenv('DB_NAME', 'magnus'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', '')
            )
            cur = conn.cursor()

            # Map common keys to columns
            if preference_key == 'risk_tolerance':
                cur.execute("""
                    INSERT INTO ava_user_preferences (user_id, risk_tolerance, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (user_id)
                    DO UPDATE SET risk_tolerance = %s, updated_at = NOW()
                """, (user_id, preference_value, preference_value))
            elif preference_key == 'max_position_size':
                cur.execute("""
                    INSERT INTO ava_user_preferences (user_id, max_position_size, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (user_id)
                    DO UPDATE SET max_position_size = %s, updated_at = NOW()
                """, (user_id, preference_value, preference_value))
            elif preference_key == 'preferred_strategy':
                cur.execute("""
                    INSERT INTO ava_user_preferences (user_id, preferred_strategy, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (user_id)
                    DO UPDATE SET preferred_strategy = %s, updated_at = NOW()
                """, (user_id, preference_value, preference_value))
            else:
                # Store in JSON column
                cur.execute("""
                    INSERT INTO ava_user_preferences (user_id, preferences_json, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (user_id)
                    DO UPDATE SET preferences_json = jsonb_set(
                        COALESCE(ava_user_preferences.preferences_json, '{}'::jsonb),
                        %s,
                        to_jsonb(%s)
                    ), updated_at = NOW()
                """, (user_id, json.dumps({preference_key: preference_value}),
                     f'{{{preference_key}}}', preference_value))

            conn.commit()
            conn.close()
            logger.info(f"✅ Set preference {preference_key}={preference_value} for {user_id}")
        except Exception as e:
            logger.error(f"Error setting preference: {e}")

    def get_available_watchlists(self) -> List[str]:
        """Get list of available watchlists from database"""
        try:
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                database=os.getenv('DB_NAME', 'magnus'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', '')
            )
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT name FROM watchlists WHERE active = true ORDER BY name")
            watchlists = [row[0] for row in cur.fetchall()]
            conn.close()
            return watchlists
        except Exception as e:
            logger.error(f"Error fetching watchlists: {e}")
            return []

    def get_available_tickers(self) -> List[str]:
        """Get list of tickers from database"""
        try:
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                database=os.getenv('DB_NAME', 'magnus'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', '')
            )
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT ticker FROM stocks ORDER BY ticker LIMIT 50")
            tickers = [row[0] for row in cur.fetchall()]
            conn.close()
            return tickers
        except Exception as e:
            logger.error(f"Error fetching tickers: {e}")
            return []

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

            if not results:
                return "Query executed successfully but returned no results."

            # Format results nicely
            formatted = "**Query Results:**\n\n"
            for i, row in enumerate(results[:10], 1):  # Limit to 10 rows
                formatted += f"**Row {i}:**\n"
                for key, value in dict(row).items():
                    formatted += f"- {key}: {value}\n"
                formatted += "\n"

            if len(results) > 10:
                formatted += f"\n... and {len(results) - 10} more rows"

            return formatted
        except Exception as e:
            return f"❌ Database error: {str(e)}\n\nPlease check your query syntax."

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
            return f"✅ Created task #{task_id}: **{title}**\n\nPriority: {priority}"
        except Exception as e:
            return f"❌ Error creating task: {str(e)}"

    def analyze_watchlist(self, watchlist_name: str) -> str:
        """Analyze watchlist for opportunities"""
        try:
            analyzer = WatchlistStrategyAnalyzer()
            results = analyzer.analyze_watchlist(watchlist_name, min_score=60.0)
            if results:
                summary = f"**Found {len(results)} opportunities in {watchlist_name}:**\n\n"
                for i, r in enumerate(results[:5], 1):  # Top 5
                    summary += f"{i}. **{r['ticker']}**: Score {r['score']:.0f}/100\n"
                    if 'premium_yield' in r:
                        summary += f"   - Premium Yield: {r['premium_yield']:.2%}\n"
                summary += f"\n💡 Type 'analyze {results[0]['ticker']}' for detailed analysis"
                return summary
            return f"No opportunities found in {watchlist_name} meeting minimum criteria (60+ score)"
        except Exception as e:
            return f"❌ Error analyzing watchlist: {str(e)}"

    def get_portfolio_status(self) -> str:
        """Get portfolio status - PHASE 1: Show data directly from database"""
        try:
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                database=os.getenv('DB_NAME', 'magnus'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', '')
            )

            # Get latest portfolio balance
            cur = conn.cursor()
            cur.execute("""
                SELECT cash, options_value, total_value, date
                FROM portfolio_balances
                ORDER BY date DESC
                LIMIT 1
            """)
            balance_row = cur.fetchone()

            # Get active positions count
            cur.execute("SELECT COUNT(*) FROM positions WHERE status = 'open'")
            open_positions = cur.fetchone()[0] if cur.fetchone() else 0

            conn.close()

            if balance_row:
                cash, options_val, total_val, date = balance_row
                return f"""📊 **Portfolio Summary:**

💰 **Cash:** ${cash:,.2f}
📈 **Options Value:** ${options_val:,.2f}
💼 **Total Value:** ${total_val:,.2f}
📅 **As of:** {date}

📍 **Active Positions:** {open_positions}

💡 *Ask me about specific positions or opportunities!*"""
            else:
                return "📊 No portfolio data found. Make sure data sync is running."

        except Exception as e:
            logger.error(f"Error fetching portfolio: {e}")
            # Fallback to Robinhood if database fails
            try:
                import robin_stocks.robinhood as rh
                username = os.getenv('ROBINHOOD_USERNAME')
                password = os.getenv('ROBINHOOD_PASSWORD')
                if username and password:
                    rh.login(username, password)
                    account = rh.profiles.load_account_profile()
                    portfolio_value = float(account.get('portfolio_cash', 0))
                    return f"💼 **Portfolio Value:** ${portfolio_value:,.2f}"
            except:
                pass
            return f"⚠️ Unable to fetch portfolio: {str(e)}"

    def get_stock_price(self, ticker: str) -> str:
        """Get current stock price"""
        try:
            import robin_stocks.robinhood as rh
            quote = rh.get_quotes(ticker)
            if quote and len(quote) > 0:
                price = float(quote[0].get('last_trade_price', 0))
                change = float(quote[0].get('change', 0))
                change_pct = float(quote[0].get('change_percentage', 0))

                emoji = "🟢" if change >= 0 else "🔴"
                return f"{emoji} **{ticker}**: ${price:.2f} ({change:+.2f}, {change_pct:+.2f}%)"
            return f"❌ Unable to get price for {ticker}"
        except Exception as e:
            return f"❌ Error getting stock price: {str(e)}"

    def extract_ticker(self, text: str) -> Optional[str]:
        """Extract ticker symbol from text"""
        # Look for 1-5 letter uppercase words
        words = text.upper().split()
        for word in words:
            # Remove common prefixes/suffixes
            clean_word = re.sub(r'[^A-Z]', '', word)
            if 1 <= len(clean_word) <= 5 and clean_word.isalpha():
                return clean_word
        return None

    def ask_clarifying_question(self, intent: str, context: Dict = None) -> Dict:
        """Generate intelligent clarifying questions based on intent"""

        if intent == "analyze_watchlist":
            available_watchlists = self.get_available_watchlists()
            if available_watchlists:
                question = "**Which watchlist would you like to analyze?**\n\n"
                question += "Available watchlists:\n"
                for i, wl in enumerate(available_watchlists[:10], 1):
                    question += f"{i}. {wl}\n"
                question += "\nJust type the name or number."

                return {
                    'question': question,
                    'state': ConversationState.AWAITING_WATCHLIST_NAME,
                    'suggestions': available_watchlists[:10],
                    'context': {'available_options': available_watchlists}
                }
            else:
                return {
                    'question': "Please specify a watchlist name (e.g., 'NVDA', 'TECH', 'WHEEL')",
                    'state': ConversationState.AWAITING_WATCHLIST_NAME,
                    'suggestions': ['NVDA', 'AAPL', 'TECH', 'WHEEL'],
                    'context': {}
                }

        elif intent == "stock_price":
            available_tickers = self.get_available_tickers()
            question = "**Which stock price would you like to check?**\n\n"
            if available_tickers:
                question += "Popular tickers:\n"
                question += ", ".join(available_tickers[:15])
            question += "\n\nJust type the ticker symbol (e.g., AAPL, TSLA, NVDA)"

            return {
                'question': question,
                'state': ConversationState.AWAITING_TICKER_SYMBOL,
                'suggestions': available_tickers[:15] if available_tickers else ['AAPL', 'TSLA', 'NVDA'],
                'context': {}
            }

        elif intent == "create_task":
            if not context or not context.get('title'):
                return {
                    'question': "**What would you like the task to be about?**\n\nExamples:\n- Improve dashboard performance\n- Add earnings calendar feature\n- Fix positions page bug",
                    'state': ConversationState.AWAITING_TASK_DETAILS,
                    'suggestions': [],
                    'context': {}
                }
            else:
                # Have title, ask for priority
                return {
                    'question': f"**Task:** {context['title']}\n\nWhat priority should this be?\n\n1. 🔴 High\n2. 🟡 Medium\n3. 🟢 Low",
                    'state': ConversationState.AWAITING_TASK_PRIORITY,
                    'suggestions': ['high', 'medium', 'low'],
                    'context': context
                }

        elif intent == "database_query":
            return {
                'question': "**What database query would you like to run?**\n\nExamples:\n- `SELECT * FROM watchlists LIMIT 5`\n- `SELECT COUNT(*) FROM positions`\n- `SELECT ticker, score FROM opportunities ORDER BY score DESC LIMIT 10`\n\nPlease provide your SQL query:",
                'state': ConversationState.AWAITING_SQL_QUERY,
                'suggestions': [],
                'context': {}
            }

        return {
            'question': "I'm not sure what you need. Could you provide more details?",
            'state': ConversationState.IDLE,
            'suggestions': [],
            'context': {}
        }

    def query_with_rag_and_llm(self, user_query: str, conversation_history: List[Dict] = None, user_id: str = "web_user") -> Dict:
        """PHASE 1 & 3: Use RAG + LLM for intelligent responses with multi-turn context"""
        try:
            # Step 1: Query RAG for relevant context
            rag_context = ""
            if self.rag_service:
                try:
                    rag_results = self.rag_service.query(user_query, top_k=3)
                    if rag_results and 'documents' in rag_results:
                        rag_context = "\n\n".join(rag_results['documents'][:3])
                        logger.info(f"✅ RAG retrieved {len(rag_results['documents'])} documents")
                except Exception as e:
                    logger.warning(f"RAG query failed: {e}")

            # PHASE 3: Step 2 - Build comprehensive conversation history (last 5 messages)
            history_text = ""
            if conversation_history and len(conversation_history) > 0:
                # Get last 5 messages for context
                recent_messages = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history

                # Build formatted history
                history_parts = []
                for msg in recent_messages:
                    role = "User" if msg.get('role') == 'user' else "AVA"
                    content = msg.get('content', '')
                    history_parts.append(f"{role}: {content}")

                history_text = "\n".join(history_parts)
                logger.info(f"✅ Using {len(recent_messages)} messages for context")

            # PHASE 3: Get user preferences for personalization
            user_prefs = self.get_user_preferences(user_id)
            prefs_text = f"""User preferences:
- Risk Tolerance: {user_prefs['risk_tolerance']}
- Favorite Tickers: {', '.join(user_prefs['favorite_tickers']) if user_prefs['favorite_tickers'] else 'None set'}
- Max Position Size: ${user_prefs['max_position_size']:,}
- Preferred Strategy: {user_prefs['preferred_strategy']}"""

            # PHASE 3: Step 3 - Enhanced prompt with better context fusion
            prompt = f"""You are AVA, an intelligent trading assistant for the Magnus platform.

{prefs_text}

Context from knowledge base:
{rag_context if rag_context else "No specific knowledge base context available."}

Recent conversation history:
{history_text if history_text else "This is the first message in this conversation."}

Current user question: {user_query}

Instructions:
- Reference previous conversation context when relevant (e.g., "As I mentioned earlier...")
- If this is a follow-up question, connect it to the previous topic
- Provide helpful, accurate responses about trading, portfolio management, and the Magnus platform
- If you're not confident, include your confidence level (e.g., "I'm 70% confident that...")
- Be conversational and friendly - you're having a dialogue, not just answering isolated questions
- Keep responses concise but informative
- If the user asks about portfolio, positions, or opportunities, tell them you can show that data

Response:"""

            # Step 4: Generate response with LLM (check selected model)
            confidence = 0.8  # Default confidence
            selected_model = st.session_state.get('selected_model', 'Qwen 2.5 32B (Local)')

            # Route to local LLM if Qwen model is selected
            if 'Qwen' in selected_model and 'Local' in selected_model and self.local_llm:
                try:
                    logger.info(f"Using local LLM: {selected_model}")
                    # Determine complexity based on selected model
                    if 'Coder' in selected_model:
                        complexity = TaskComplexity.CODING
                    elif '14B' in selected_model:
                        complexity = TaskComplexity.FAST
                    else:
                        complexity = TaskComplexity.BALANCED
                    response_text = self.local_llm.query(
                        prompt=prompt,
                        complexity=complexity,
                        use_trading_context=True,
                        max_tokens=2000  # Increased from 500 for fuller responses
                    )
                    logger.info("✅ Local LLM generated response successfully")

                    # PHASE 3: Parse confidence from response if LLM included it
                    confidence_match = re.search(r"(\d+)%\s+confident", response_text.lower())
                    if confidence_match:
                        confidence = float(confidence_match.group(1)) / 100.0
                        logger.info(f"✅ Parsed confidence: {confidence*100:.0f}%")

                    return {
                        'text': response_text.strip(),
                        'confidence': confidence,
                        'used_rag': bool(rag_context),
                        'used_history': bool(history_text)
                    }
                except Exception as e:
                    logger.warning(f"Local LLM generation failed: {e}, falling back to cloud LLM")

            # Use cloud LLM service (Groq, Gemini, etc.)
            if self.llm_service:
                try:
                    response = self.llm_service.generate(
                        prompt=prompt,
                        provider=None,  # Auto-select FREE provider (Groq/Llama)
                        use_cache=True,
                        max_tokens=500
                    )
                    if response and 'text' in response:
                        logger.info("✅ LLM generated response successfully")
                        response_text = response['text'].strip()

                        # PHASE 3: Parse confidence from response if LLM included it
                        confidence_match = re.search(r"(\d+)%\s+confident", response_text.lower())
                        if confidence_match:
                            confidence = float(confidence_match.group(1)) / 100.0
                            logger.info(f"✅ Parsed confidence: {confidence*100:.0f}%")

                        return {
                            'text': response_text,
                            'confidence': confidence,
                            'used_rag': bool(rag_context),
                            'used_history': bool(history_text)
                        }
                except Exception as e:
                    logger.warning(f"LLM generation failed: {e}")

            # Fallback: Basic response if RAG/LLM fails
            if rag_context:
                return {
                    'text': f"Based on what I know:\n\n{rag_context[:500]}...\n\n💡 *Ask me for more details!*",
                    'confidence': 0.6,
                    'used_rag': True,
                    'used_history': False
                }
            else:
                # More helpful fallback with specific suggestions
                suggestions = self._get_helpful_suggestions(user_query)
                return {
                    'text': suggestions,
                    'confidence': 0.4,
                    'used_rag': False,
                    'used_history': False
                }

        except Exception as e:
            logger.error(f"Error in RAG+LLM query: {e}")
            return {
                'text': f"I encountered an error processing your request. Please try rephrasing or type 'help'.",
                'confidence': 0.2,
                'used_rag': False,
                'used_history': False
            }

    def _get_helpful_suggestions(self, user_query: str) -> str:
        """Generate helpful suggestions based on the user's query"""
        query_lower = user_query.lower()

        # Categorize the query and provide relevant suggestions
        if any(word in query_lower for word in ['option', 'put', 'call', 'strike', 'premium', 'wheel', 'covered', 'spread']):
            return f"""I'd be happy to help with options! Here's what I can do:

**Options Analysis:**
• "What is the wheel strategy?" - Learn about this income strategy
• "Explain covered calls" - Understand covered call mechanics
• "What are the Greeks?" - Learn about delta, theta, gamma
• "Find CSP opportunities" - Scan for cash-secured put opportunities

**For specific analysis:**
• "Analyze AAPL options" - Get options analysis for a stock
• "Show my positions" - See your current option positions

*What would you like to explore?*"""

        elif any(word in query_lower for word in ['portfolio', 'position', 'holding', 'balance', 'account']):
            return f"""I can help with your portfolio! Try asking:

• "Show my portfolio" - See your current holdings
• "How is my portfolio doing?" - Get a performance summary
• "What positions do I have?" - List all open positions
• "Show my balance" - See account balance
• "Analyze my positions" - Get AI recommendations

*Which would you like to see?*"""

        elif any(word in query_lower for word in ['stock', 'price', 'ticker', 'symbol', 'quote']):
            return f"""I can help with stock information! Try:

• "What's AAPL at?" - Get current price
• "Show NVDA chart" - View price chart
• "Analyze TSLA" - Get technical analysis
• "Add MSFT to watchlist" - Track a stock

*What stock would you like to know about?*"""

        elif any(word in query_lower for word in ['risk', 'safe', 'dangerous', 'protect', 'hedge']):
            return f"""I can help with risk management! Here's what I know:

• "What is position sizing?" - Learn how to size trades
• "Explain max loss rules" - Understand risk limits
• "How to manage drawdowns" - Recovery strategies
• "Portfolio diversification" - Spread your risk

*What aspect of risk management interests you?*"""

        elif any(word in query_lower for word in ['technical', 'chart', 'indicator', 'support', 'resistance', 'trend']):
            return f"""I can help with technical analysis! Try asking:

• "What is RSI?" - Learn about indicators
• "Explain support and resistance" - Key price levels
• "How to read Fibonacci levels" - Retracement analysis
• "What are chart patterns?" - Visual trading signals

*What technical concept would you like to learn?*"""

        elif any(word in query_lower for word in ['personality', 'style', 'mode', 'change ava']):
            return f"""I can change my communication style! Available personalities:

• **Friendly** (default) - Warm and approachable
• **Professional** - Formal and data-focused
• **Witty** - Humorous market commentary
• **Mentor** - Educational and patient
• **Coach** - Motivational and energizing
• **Analyst** - Bloomberg terminal style
• **Guru** - Zen master wisdom

Use /personality <name> or ask "Change to coach mode"!"""

        else:
            # Generic helpful response
            return f"""I'm here to help with your trading! Here are some things I can do:

**Market Analysis:**
• Get stock prices and charts
• Technical analysis and indicators
• Options opportunities and Greeks

**Portfolio:**
• View positions and balances
• AI-powered recommendations
• Performance tracking

**Learning:**
• Options strategies (wheel, spreads, etc.)
• Risk management principles
• Technical analysis concepts

**Try asking:**
• "What's my portfolio worth?"
• "Find wheel opportunities"
• "Explain the wheel strategy"
• "Show AAPL price"

*What would you like help with?*"""

    def process_message(self, user_message: str, user_id: str = "web_user", platform: str = "web") -> Dict:
        """Process user message with intelligent question-asking"""

        # PHASE 2: Start response time tracking
        start_time = time.time()

        # Initialize session state for conversation tracking
        if 'ava_state' not in st.session_state:
            st.session_state.ava_state = ConversationState.IDLE
            st.session_state.ava_context = {}

        # Start or get conversation
        conversation_id = self.memory_manager.get_active_conversation(user_id, platform)
        if not conversation_id:
            conversation_id = self.memory_manager.start_conversation(user_id, platform)

        # Ensure user_message is a string (handle ChatInputValue or other objects)
        user_message = str(user_message) if user_message else ""
        message_lower = user_message.lower()
        response = None
        intent = "unknown"
        action = None
        success = True
        needs_clarification = False
        confidence_score = 0.8  # PHASE 2: Default confidence

        try:
            # Check if we're in the middle of a multi-turn conversation
            current_state = st.session_state.ava_state

            if current_state == ConversationState.AWAITING_WATCHLIST_NAME:
                # User is responding to watchlist question
                intent = "analyze_watchlist"

                # Check if it's a number (selecting from list)
                if user_message.strip().isdigit():
                    idx = int(user_message.strip()) - 1
                    available = st.session_state.ava_context.get('available_options', [])
                    if 0 <= idx < len(available):
                        watchlist_name = available[idx]
                    else:
                        response = f"Invalid selection. Please choose 1-{len(available)}"
                        success = False
                else:
                    watchlist_name = user_message.strip()

                if success:
                    response = self.analyze_watchlist(watchlist_name)
                    action = "watchlist_analyzed"
                    st.session_state.ava_state = ConversationState.IDLE
                    st.session_state.ava_context = {}

            elif current_state == ConversationState.AWAITING_TICKER_SYMBOL:
                # User is responding to ticker question
                intent = "stock_price"
                ticker = self.extract_ticker(user_message)
                if ticker:
                    response = self.get_stock_price(ticker)
                    action = "stock_price_fetched"
                    st.session_state.ava_state = ConversationState.IDLE
                else:
                    response = "❌ Please provide a valid ticker symbol (e.g., AAPL, TSLA)"
                    success = False

            elif current_state == ConversationState.AWAITING_TASK_DETAILS:
                # User is providing task details
                intent = "create_task"
                st.session_state.ava_context['title'] = user_message

                # Ask for priority
                clarification = self.ask_clarifying_question("create_task", st.session_state.ava_context)
                response = clarification['question']
                st.session_state.ava_state = clarification['state']
                needs_clarification = True

            elif current_state == ConversationState.AWAITING_TASK_PRIORITY:
                # User is providing priority
                intent = "create_task"
                priority_map = {'1': 'high', '2': 'medium', '3': 'low', 'high': 'high', 'medium': 'medium', 'low': 'low'}
                priority = priority_map.get(user_message.strip().lower(), 'medium')

                title = st.session_state.ava_context.get('title', 'User requested task')
                response = self.create_task(title, f"User request: {title}", priority)
                action = "task_created"
                st.session_state.ava_state = ConversationState.IDLE
                st.session_state.ava_context = {}

            elif current_state == ConversationState.AWAITING_SQL_QUERY:
                # User is providing SQL query
                intent = "database_query"
                if 'select' in message_lower:
                    response = self.query_database(user_message)
                    action = "database_query_executed"
                    st.session_state.ava_state = ConversationState.IDLE
                else:
                    response = "❌ Please provide a valid SELECT query"
                    success = False

            else:
                # New conversation - detect intent

                # AUTONOMOUS TASK SYSTEM - Handle "task:" commands first
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
                        query_start = message_lower.find('select')
                        query = user_message[query_start:]
                        response = self.query_database(query)
                        action = "database_query_executed"
                    else:
                        clarification = self.ask_clarifying_question("database_query")
                        response = clarification['question']
                        st.session_state.ava_state = clarification['state']
                        needs_clarification = True

                # Task creation
                elif any(word in message_lower for word in ['create task', 'add task', 'new task', 'improve']):
                    intent = "create_task"
                    task_title = user_message.replace('create task', '').replace('add task', '').replace('new task', '').replace('improve', '').strip()

                    if task_title and len(task_title) > 5:
                        # Have enough info, create directly
                        response = self.create_task(task_title, f"User request: {user_message}", "medium")
                        action = "task_created"
                    else:
                        # Ask for more details
                        clarification = self.ask_clarifying_question("create_task")
                        response = clarification['question']
                        st.session_state.ava_state = clarification['state']
                        needs_clarification = True

                # Watchlist analysis
                elif any(word in message_lower for word in ['analyze', 'watchlist', 'opportunities']):
                    intent = "analyze_watchlist"
                    watchlist_name = None

                    # Try to extract watchlist name
                    for word in message_lower.split():
                        if len(word) >= 3 and word.isalpha():
                            watchlist_name = word.upper()
                            break

                    if watchlist_name:
                        response = self.analyze_watchlist(watchlist_name)
                        action = "watchlist_analyzed"
                    else:
                        clarification = self.ask_clarifying_question("analyze_watchlist")
                        response = clarification['question']
                        st.session_state.ava_state = clarification['state']
                        st.session_state.ava_context = clarification.get('context', {})
                        needs_clarification = True

                # Portfolio status
                elif any(word in message_lower for word in ['portfolio', 'balance', 'account']):
                    intent = "portfolio_status"
                    response = self.get_portfolio_status()
                    action = "portfolio_checked"

                # Stock price
                elif any(word in message_lower for word in ['price of', 'stock price', 'what is', "what's"]):
                    intent = "stock_price"
                    ticker = self.extract_ticker(user_message)

                    if ticker:
                        response = self.get_stock_price(ticker)
                        action = "stock_price_fetched"
                    else:
                        clarification = self.ask_clarifying_question("stock_price")
                        response = clarification['question']
                        st.session_state.ava_state = clarification['state']
                        needs_clarification = True

                # General help
                elif any(word in message_lower for word in ['help', 'what can you', 'how do i']):
                    intent = "help"
                    response = """**I can help you with:**

📊 **Database Queries:**
- "Show me all pending tasks"
- "Query: SELECT * FROM watchlists"

✅ **Task Management:**
- "Create task to improve dashboard"
- "Add task for better analysis"

📈 **Watchlist Analysis:**
- "Analyze NVDA watchlist"
- "Show opportunities in TECH"

💼 **Portfolio:**
- "What's my portfolio balance?"
- "Check my account"

💰 **Stock Prices:**
- "What's the price of AAPL?"
- "Stock price TSLA"

ℹ️ **Magnus Information:**
- "What is Magnus?"
- "How does Magnus work?"

💡 **Tip:** I'll ask clarifying questions if I need more information!"""
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

I'm **AVA**, your AI assistant. I can help you analyze positions, create tasks, query data, and more!

💡 **I'm here to help!** Just ask me anything, and I'll guide you with clarifying questions if needed."""
                    action = "about_provided"

                # PHASE 3: Default response - Use RAG + LLM for general queries
                else:
                    intent = "general_conversation"

                    # Use RAG + LLM with conversation history and user preferences
                    llm_result = self.query_with_rag_and_llm(
                        user_query=user_message,
                        conversation_history=st.session_state.get('ava_messages', []),
                        user_id=user_id
                    )

                    response = llm_result['text']
                    confidence_score = llm_result['confidence']
                    success = confidence_score > 0.5  # Success if reasonably confident

                    # PHASE 3: Add proactive suggestions based on response context
                    if any(word in response.lower() for word in ['portfolio', 'balance', 'account']):
                        response += "\n\n💡 **Would you like me to:**\n- Find new opportunities\n- Analyze current positions\n- Check earnings calendar"
                    elif any(word in response.lower() for word in ['opportunity', 'opportunities', 'find']):
                        response += "\n\n💡 **I can also help:**\n- Analyze specific watchlists\n- Screen for high-premium CSPs\n- Check technical indicators"
                    elif any(word in response.lower() for word in ['position', 'positions', 'trade']):
                        response += "\n\n💡 **Related actions:**\n- View all active positions\n- Get AI recommendations\n- Calculate risk metrics"

                    logger.info(f"✅ RAG+LLM response: confidence={confidence_score:.2f}, "
                               f"used_rag={llm_result['used_rag']}, "
                               f"used_history={llm_result['used_history']}")

                    # Log as unanswered only if very low confidence
                    if confidence_score < 0.4:
                        self.memory_manager.record_unanswered_question(
                            user_question=user_message,
                            intent_detected=intent,
                            confidence_score=confidence_score,
                            failure_reason="low_confidence",
                            conversation_id=conversation_id,
                            context={'platform': platform}
                        )

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            response = f"❌ I encountered an error: {str(e)}\n\nPlease try again or type 'help' for assistance."
            success = False

        # Apply personality styling to response
        if response and self.personality:
            try:
                # Detect emotional context from data
                data_context = {
                    'confidence': confidence_score,
                    'success': success,
                    'intent': intent
                }
                emotional_state = self.personality.detect_emotional_context(data_context)
                self.personality.set_emotional_state(emotional_state)

                # Apply personality styling
                response = self.personality.style_response(response, data_context)
                logger.debug(f"Applied personality: {self.personality.mode.value}, emotion: {emotional_state.value}")
            except Exception as e:
                logger.warning(f"Personality styling failed: {e}")

        # PHASE 2: Calculate response time
        elapsed_time = time.time() - start_time

        # Log the interaction
        self.memory_manager.log_message(
            conversation_id=conversation_id,
            user_message=user_message,
            ava_response=response,
            intent_detected=intent,
            confidence_score=confidence_score if success else 0.2,
            action_performed=action,
            action_success=success,
            model_used='enhanced_ava'
        )

        # PHASE 2: Log slow responses
        if elapsed_time > 2.0:
            logger.warning(f"⚠️ Slow response: {elapsed_time:.2f}s (target: <2s)")

        return {
            'response': response,
            'success': success,
            'intent': intent,
            'conversation_id': conversation_id,
            'needs_clarification': needs_clarification,
            'state': st.session_state.ava_state.value if hasattr(st.session_state, 'ava_state') else 'idle',
            'response_time': elapsed_time,  # PHASE 2: Include timing
            'confidence': confidence_score  # PHASE 2: Include confidence
        }


def show_enhanced_ava(key_prefix=""):
    """
    Display Enhanced AVA with modern chat interface
    Redesigned for better UX: larger chat area, integrated input, cleaner layout

    Args:
        key_prefix: Optional prefix for widget keys to avoid conflicts when used on multiple pages
    """

    # Initialize Enhanced AVA in session state
    if 'enhanced_ava' not in st.session_state:
        st.session_state.enhanced_ava = EnhancedAVA()

    # Initialize message history
    if 'ava_messages' not in st.session_state:
        st.session_state.ava_messages = []

    # Initialize conversation state
    if 'ava_state' not in st.session_state:
        st.session_state.ava_state = ConversationState.IDLE
        st.session_state.ava_context = {}

    # Modern Chat UI CSS - ChatGPT/Claude-inspired
    st.markdown("""
        <style>
        /* Main chat container - transparent, no gray box */
        .chat-container {
            background: transparent !important;
            border-radius: 0;
            padding: 0;
            margin-bottom: 10px;
            box-shadow: none !important;
            min-height: 0;
            max-height: 500px;
            overflow-y: auto;
            border: none !important;
        }

        /* Ultra compact model selector - very small */
        [data-testid="stSelectbox"] {
            max-height: 9px !important;
        }

        [data-testid="stSelectbox"] > div {
            min-height: 9px !important;
            height: 9px !important;
        }

        [data-testid="stSelectbox"] input {
            min-height: 9px !important;
            height: 9px !important;
            padding: 0px 4px !important;
            font-size: 9px !important;
        }

        /* Compact buttons for image controls */
        [data-testid="stButton"] button {
            padding: 4px 8px !important;
            font-size: 12px !important;
            min-height: 28px !important;
        }

        /* AVA image column */
        .ava-image-container {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 10px;
        }

        /* Chat message bubbles - no borders */
        .user-message-bubble {
            background: #2563eb;
            color: white;
            padding: 8px 12px;
            border-radius: 14px;
            margin: 4px 0;
            margin-left: auto;
            max-width: 80%;
            width: fit-content;
            float: right;
            clear: both;
            font-size: 14px;
            line-height: 1.4;
            border: none !important;
        }

        .ava-message-bubble {
            background: #9ca3af;
            color: #1f2937;
            padding: 8px 12px;
            border-radius: 14px;
            margin: 4px 0;
            margin-right: auto;
            max-width: 80%;
            width: fit-content;
            box-shadow: none !important;
            float: left;
            clear: both;
            font-size: 14px;
            line-height: 1.4;
            border: none !important;
        }

        /* Input area container - transparent */
        .input-container {
            background: transparent;
            padding: 0;
            margin-top: 10px;
        }

        .input-container:focus-within {
            /* No special styling needed */
        }

        /* Text input styling - no label, seamless */
        .stTextInput > div > div > input {
            border: none !important;
            background: transparent !important;
            padding: 12px 16px !important;
            font-size: 15px !important;
            box-shadow: none !important;
        }

        .stTextInput > div > div > input:focus {
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }

        /* Hide text input label completely */
        .stTextInput > label {
            display: none !important;
        }

        .stTextInput {
            margin-bottom: 0 !important;
        }

        /* Send button - integrated into input */
        .send-button-container button {
            background: #667eea !important;
            color: white !important;
            border: none !important;
            border-radius: 20px !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            transition: all 0.2s ease !important;
            box-shadow: none !important;
        }

        .send-button-container button:hover {
            background: #5a67d8 !important;
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }

        /* Quick action buttons - modern style */
        .quick-action-btn button {
            background: white !important;
            color: #374151 !important;
            border: 2px solid #e5e7eb !important;
            border-radius: 10px !important;
            padding: 10px 16px !important;
            font-weight: 500 !important;
            font-size: 14px !important;
            transition: all 0.2s ease !important;
        }

        .quick-action-btn button:hover {
            border-color: #667eea !important;
            color: #667eea !important;
            background: #f0f4ff !important;
        }

        /* Expander styling - minimal */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            border-radius: 10px !important;
            padding: 12px 16px !important;
            font-size: 16px !important;
        }

        /* Remove margins and padding for cleaner look */
        .element-container {
            margin-bottom: 0 !important;
        }

        /* Column padding adjustments */
        [data-testid="column"] {
            padding: 0 8px !important;
        }

        /* Feedback buttons - subtle */
        .feedback-btn button {
            background: transparent !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 8px !important;
            padding: 6px 12px !important;
            font-size: 13px !important;
            color: #6b7280 !important;
        }

        .feedback-btn button:hover {
            border-color: #667eea !important;
            color: #667eea !important;
        }

        /* Meta info styling - more compact */
        .stCaption {
            color: #9ca3af !important;
            font-size: 11px !important;
            margin-top: -2px !important;
            margin-bottom: 2px !important;
        }

        /* Scrollbar styling for chat */
        .chat-container::-webkit-scrollbar {
            width: 6px;
        }

        .chat-container::-webkit-scrollbar-track {
            background: #e5e7eb;
            border-radius: 10px;
        }

        .chat-container::-webkit-scrollbar-thumb {
            background: #9ca3af;
            border-radius: 10px;
        }

        .chat-container::-webkit-scrollbar-thumb:hover {
            background: #6b7280;
        }

        /* Position expander at top-left - full width, no centering */
        [data-testid="stExpander"] {
            max-width: 100%;
            margin-left: 0 !important;
            margin-right: 0 !important;
            margin-top: 0 !important;
        }

        .streamlit-expanderHeader {
            max-width: 100%;
            margin-left: 0 !important;
            margin-right: 0 !important;
        }

        /* Expander content - full width */
        [data-testid="stExpander"] > div {
            max-width: 100%;
            margin-left: 0;
            margin-right: 0;
        }

        /* Make expander header more compact and left-aligned */
        .streamlit-expanderHeader p {
            font-size: 18px !important;
            font-weight: 600 !important;
            margin: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main AVA expander - compact and at top
    with st.expander("🤖 AVA - Your Expert Trading Assistant", expanded=True):

        # Top row: AVA image on left, chat on right
        # AVA image with hourly rotation + manual control
        from pathlib import Path

        # Get all images from docs/ava/AnnaA folder
        image_folder = Path("docs/ava/AnnaA")
        image_files = sorted([f for f in image_folder.glob("*") if f.is_file() and f.suffix.lower() in ['.jpg', '.png', '.webp', '.avif', '.jpeg']])

        # Fallback to default if no images found
        if not image_files:
            image_files = [Path("docs/ava/AnnaA/AvaMain.jpg")]

        # Initialize manual rotation state
        if 'ava_manual_image_index' not in st.session_state:
            st.session_state.ava_manual_image_index = None
        if 'ava_manual_mode' not in st.session_state:
            st.session_state.ava_manual_mode = False

        # Determine which image to show
        if st.session_state.ava_manual_mode and st.session_state.ava_manual_image_index is not None:
            # Use manually selected image
            image_index = st.session_state.ava_manual_image_index % len(image_files)
        else:
            # Use automatic time-based rotation (changes every 15 minutes)
            now = datetime.now()
            interval_15min = (now.hour * 4) + (now.minute // 15)
            image_index = interval_15min % len(image_files)

        avatar_path = image_files[image_index]

        # Main layout: Image column (left) and Control column (right)
        img_col, control_col = st.columns([2, 3])

        with img_col:
            # Display rotating image (spans full height)
            if avatar_path.exists():
                st.image(str(avatar_path), use_container_width=True)
            else:
                try:
                    st.image("assets/ava_avatar.png", width=600)
                except:
                    st.markdown("### 🤖 AVA")

        with control_col:
            # Top: Image rotation buttons + Model selector + Icon buttons (all in one row)
            icon_row = st.columns([1.5, 0.3, 0.3, 0.3, 5.5, 0.5, 0.5, 0.5, 0.5, 0.5])

            # Image rotation buttons
            with icon_row[1]:
                if st.button("◀️", key=f"{key_prefix}prev_image", help="Previous image"):
                    st.session_state.ava_manual_mode = True
                    current = st.session_state.ava_manual_image_index if st.session_state.ava_manual_image_index is not None else image_index
                    st.session_state.ava_manual_image_index = (current - 1) % len(image_files)
                    st.rerun()
            with icon_row[2]:
                if st.button("🔄", key=f"{key_prefix}auto_rotate", help="Auto rotate"):
                    st.session_state.ava_manual_mode = False
                    st.session_state.ava_manual_image_index = None
                    st.rerun()
            with icon_row[3]:
                if st.button("▶️", key=f"{key_prefix}next_image", help="Next image"):
                    st.session_state.ava_manual_mode = True
                    current = st.session_state.ava_manual_image_index if st.session_state.ava_manual_image_index is not None else image_index
                    st.session_state.ava_manual_image_index = (current + 1) % len(image_files)
                    st.rerun()

            # Model selector (between rotation and icon buttons)
            with icon_row[4]:
                selected_model = st.selectbox(
                    "model",
                    options=[
                        "Groq (Llama 3.3 70B)",
                        "Gemini 2.5 Pro",
                        "DeepSeek Chat",
                        "GPT-4 Turbo",
                        "Claude Sonnet 3.5",
                        "Qwen 2.5 32B (Local)",
                        "Qwen 2.5 Coder 32B (Local - Coding)",
                        "Qwen 2.5 14B (Local - Fast)"
                    ],
                    index=5,
                    key=f"{key_prefix}model_sel_top",
                    label_visibility="collapsed"
                )
                st.session_state.selected_model = selected_model

            # Icon buttons
            with icon_row[5]:
                if st.button("➕", key=f"{key_prefix}new_chat_btn_top", help="New chat"):
                    st.session_state.ava_messages = []
                    st.rerun()
            with icon_row[6]:
                if st.button("🗑️", key=f"{key_prefix}clear_chat_top", help="Clear chat"):
                    st.session_state.ava_messages = []
                    st.rerun()
            with icon_row[7]:
                if st.button("🎭", key=f"{key_prefix}personality_btn_top", help="Personality"):
                    st.session_state.show_personality = not st.session_state.get('show_personality', False)
                    st.rerun()
            with icon_row[8]:
                if st.button("🎤", key=f"{key_prefix}voice_btn_top", help="Voice"):
                    st.session_state.show_voice = not st.session_state.get('show_voice', False)
                    st.rerun()
            with icon_row[9]:
                if st.button("⚙️", key=f"{key_prefix}settings_btn_top", help="Settings"):
                    st.session_state.show_settings = not st.session_state.get('show_settings', False)
                    st.rerun()

            # SECTION 1: Conversation History
            st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
            if st.session_state.ava_messages:
                # Wrap messages in chat container (transparent background)
                st.markdown('<div class="chat-container">', unsafe_allow_html=True)

                # Display last 10 messages for better context
                for idx, msg in enumerate(st.session_state.ava_messages[-10:]):
                    if msg['role'] == 'user':
                        st.markdown(f'<div class="user-message-bubble">👤 {msg["content"]}</div>', unsafe_allow_html=True)
                        st.markdown('<div style="clear: both;"></div>', unsafe_allow_html=True)
                    else:
                        # AVA response
                        response_text = msg['content']
                        confidence = msg.get('confidence', 0.8)
                        response_time = msg.get('response_time', 0.0)

                        st.markdown(f'<div class="ava-message-bubble">🤖 {response_text}</div>', unsafe_allow_html=True)
                        st.markdown('<div style="clear: both;"></div>', unsafe_allow_html=True)

                        # Meta info below message
                        meta_info = []
                        if response_time > 0:
                            time_emoji = "⚡" if response_time < 1.0 else ("⏱️" if response_time < 2.0 else "⚠️")
                            meta_info.append(f"{time_emoji} {response_time:.2f}s")

                        if confidence < 0.8:
                            conf_emoji = "💎" if confidence >= 0.9 else ("💡" if confidence >= 0.7 else "🤔")
                            meta_info.append(f"{conf_emoji} {confidence*100:.0f}% confident")

                        if meta_info:
                            st.caption(" | ".join(meta_info), unsafe_allow_html=True)

                # Close container
                st.markdown('</div>', unsafe_allow_html=True)


            # SECTION 2: Claude-style Input Area (Everything in ONE container)
            placeholder = "How can I help you today?"

            # Initialize model selection in session state
            if 'selected_model' not in st.session_state:
                st.session_state.selected_model = "Qwen 2.5 32B (Local)"

            # Claude-style CSS - Single bordered container
            st.markdown("""
                <style>
                /* Wrapper for entire input area - no border/blur */
                .claude-wrapper {
                    background: transparent;
                    border-radius: 0;
                    padding: 8px 0;
                    border: none;
                    margin: 10px 0;
                }

                /* Single Claude-style input container */
                .stForm {
                    position: relative;
                    background: transparent !important;
                    border: none !important;
                    padding: 0 !important;
                    margin: 0 !important;
                }

                /* Hide all labels */
                .stForm label {
                    display: none !important;
                }

                /* Action buttons styling (outside form) */
                button[kind="secondary"] {
                    background: transparent !important;
                    border: none !important;
                    color: #9ca3af !important;
                    padding: 6px !important;
                    min-width: 36px !important;
                    height: 36px !important;
                    border-radius: 8px !important;
                    transition: background 0.2s !important;
                    font-size: 16px !important;
                }

                button[kind="secondary"]:hover {
                    background: rgba(255,255,255,0.1) !important;
                    color: #e5e7eb !important;
                }

                /* Textarea wrapper - contains textarea + button */
                .textarea-wrapper {
                    position: relative !important;
                    width: 100% !important;
                    display: block !important;
                }

                /* Textarea - transparent, no border, expands, with RIGHT PADDING for button */
                .stForm textarea {
                    min-height: 24px !important;
                    max-height: 300px !important;
                    padding: 8px 50px 8px 8px !important;  /* Right padding for button */
                    border: none !important;
                    background: transparent !important;
                    color: #f3f4f6 !important;
                    font-size: 15px !important;
                    line-height: 1.5 !important;
                    resize: none !important;
                    overflow-y: auto !important;
                    box-shadow: none !important;
                    width: 100% !important;
                }

                .stForm textarea:focus {
                    border: none !important;
                    outline: none !important;
                    box-shadow: none !important;
                    background: transparent !important;
                }

                .stForm textarea::placeholder {
                    color: #6b7280 !important;
                }

                /* Model selector - borderless inside form */
                .stForm .stSelectbox > div > div {
                    background: transparent !important;
                    border: none !important;
                    color: #9ca3af !important;
                    font-size: 13px !important;
                    padding: 4px 8px !important;
                }

                /* Send button - INSIDE textarea wrapper, absolute positioned */
                .textarea-wrapper button[kind="primaryFormSubmit"] {
                    position: absolute !important;
                    bottom: 8px !important;
                    right: 8px !important;
                    width: 32px !important;
                    height: 32px !important;
                    min-width: 32px !important;
                    padding: 0 !important;
                    background: #b45309 !important;
                    border: none !important;
                    border-radius: 8px !important;
                    cursor: pointer !important;
                    z-index: 10 !important;
                    transition: all 0.2s !important;
                }

                .textarea-wrapper button[kind="primaryFormSubmit"]:hover {
                    background: #92400e !important;
                }

                .textarea-wrapper button[kind="primaryFormSubmit"]:disabled {
                    background: #4a5568 !important;
                    opacity: 0.5 !important;
                }

                /* Arrow icon */
                .textarea-wrapper button[kind="primaryFormSubmit"] > div > p {
                    display: none !important;
                }

                .textarea-wrapper button[kind="primaryFormSubmit"]::before {
                    content: '↑';
                    font-size: 18px;
                    color: white;
                    font-weight: bold;
                }

                /* Hide column gaps */
                .stForm .row-widget {
                    gap: 4px !important;
                }
                </style>

                <script>
                console.log('AVA: Script loaded');

                // More aggressive repositioning with continuous monitoring
                function repositionButton() {
                    console.log('AVA: Attempting to reposition button...');

                    const form = document.querySelector('.stForm');
                    if (!form) {
                        console.log('AVA: Form not found yet');
                        return false;
                    }

                    const textarea = form.querySelector('textarea');
                    const submitBtn = form.querySelector('button[kind="primaryFormSubmit"]');

                    if (!textarea || !submitBtn) {
                        console.log('AVA: Textarea or button not found', {textarea: !!textarea, button: !!submitBtn});
                        return false;
                    }

                    // Check if already repositioned
                    if (textarea.hasAttribute('data-repositioned')) {
                        console.log('AVA: Already repositioned');
                        return true;
                    }

                    console.log('AVA: Found elements, creating wrapper...');
                    textarea.setAttribute('data-repositioned', 'true');

                    // Create wrapper
                    const wrapper = document.createElement('div');
                    wrapper.className = 'textarea-wrapper';
                    wrapper.style.cssText = 'position: relative !important; width: 100% !important; display: block !important;';

                    // Get parent of textarea
                    const textareaParent = textarea.parentElement;

                    // Insert wrapper
                    textareaParent.insertBefore(wrapper, textarea);

                    // Move elements
                    wrapper.appendChild(textarea);
                    wrapper.appendChild(submitBtn);

                    // Style textarea
                    textarea.style.cssText += 'padding-right: 50px !important; width: 100% !important;';

                    // Style button
                    submitBtn.style.cssText += 'position: absolute !important; bottom: 8px !important; right: 8px !important;';

                    console.log('AVA: Button repositioned successfully!');

                    // Auto-expand
                    function autoExpand() {
                        textarea.style.height = '24px';
                        textarea.style.height = Math.min(textarea.scrollHeight, 300) + 'px';
                    }

                    textarea.addEventListener('input', autoExpand);
                    textarea.addEventListener('keydown', function(e) {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            if (submitBtn && textarea.value.trim()) submitBtn.click();
                        }
                    });

                    autoExpand();
                    return true;
                }

                // Try immediately
                setTimeout(repositionButton, 100);

                // Try repeatedly until successful
                let attempts = 0;
                const interval = setInterval(function() {
                    attempts++;
                    if (repositionButton() || attempts > 50) {
                        clearInterval(interval);
                        console.log('AVA: Stopped trying after', attempts, 'attempts');
                    }
                }, 200);

                // Also watch for DOM changes
                const observer = new MutationObserver(function() {
                    const textarea = document.querySelector('.stForm textarea');
                    if (textarea && !textarea.hasAttribute('data-repositioned')) {
                        repositionButton();
                    }
                });

                observer.observe(document.body, { childList: true, subtree: true });
                </script>
            """, unsafe_allow_html=True)

            # Wrapper container for the entire input area
            st.markdown('<div class="claude-wrapper">', unsafe_allow_html=True)

            # Icon buttons moved to top of content area (removed from here)

            # Personality Settings Panel
            if st.session_state.get('show_personality', False) and PersonalityMode:
                with st.expander("🎭 AVA Personality", expanded=True):
                    st.markdown("### Choose AVA's Personality")

                    # Initialize personality mode in session state
                    if 'ava_personality_mode' not in st.session_state:
                        st.session_state.ava_personality_mode = PersonalityMode.FRIENDLY.value

                    # Personality mode selector
                    personality_mode = st.selectbox(
                        "Select Personality Mode",
                        options=[mode.value for mode in PersonalityMode],
                        index=list(PersonalityMode).index(PersonalityMode.FRIENDLY),
                        key=f"{key_prefix}personality_mode",
                        format_func=lambda x: {
                            'professional': '📊 Professional - Formal & Data-Focused',
                            'friendly': '😊 Friendly - Warm & Approachable',
                            'witty': '😏 Witty - Clever & Humorous',
                            'mentor': '🎓 Mentor - Teaching & Guiding',
                            'concise': '⚡ Concise - Brief & Direct',
                            'charming': '💕 Charming - Flirty & Romantic',
                            'analyst': '📈 Analyst - Bloomberg Style & Quantitative',
                            'coach': '💪 Coach - Motivational & Energizing',
                            'rebel': '🔥 Rebel - Contrarian & Independent',
                            'guru': '🙏 Guru - Zen Master & Philosophical'
                        }.get(x, x.title())
                    )

                    # Update AVA's personality
                    if st.session_state.enhanced_ava.personality:
                        new_mode = PersonalityMode(personality_mode)
                        st.session_state.enhanced_ava.personality.set_mode(new_mode)
                        st.session_state.ava_personality_mode = personality_mode

                    # Show personality description
                    if st.session_state.enhanced_ava.personality:
                        st.info(st.session_state.enhanced_ava.personality.get_personality_description())

                    # Preview greeting
                    st.markdown("### Preview Greeting")
                    if st.session_state.enhanced_ava.personality:
                        preview_greeting = st.session_state.enhanced_ava.personality.get_greeting()
                        st.markdown(f"> {preview_greeting}")

            # Voice Settings Panel
            if st.session_state.get('show_voice', False) and WebVoiceHandler:
                # Inject voice system
                WebVoiceHandler.inject_voice_controls()

                with st.expander("🎤 Voice Controls", expanded=True):
                    st.markdown("### Voice Settings")

                    # Auto-speak toggle
                    auto_speak = st.checkbox(
                        "🔊 Auto-speak AVA responses",
                        value=st.session_state.get('ava_auto_speak', False),
                        key=f"{key_prefix}auto_speak_check",
                        help="Automatically speak AVA's responses aloud"
                    )
                    st.session_state.ava_auto_speak = auto_speak

                    # Voice controls in columns
                    voice_col1, voice_col2 = st.columns(2)

                    with voice_col1:
                        st.markdown("**Speech Rate**")
                        rate = st.slider(
                            "Rate",
                            min_value=0.5,
                            max_value=2.0,
                            value=st.session_state.get('ava_voice_rate', 1.0),
                            step=0.1,
                            key=f"{key_prefix}voice_rate_slider",
                            label_visibility="collapsed"
                        )
                        st.session_state.ava_voice_rate = rate

                    with voice_col2:
                        st.markdown("**Speech Pitch**")
                        pitch = st.slider(
                            "Pitch",
                            min_value=0.5,
                            max_value=2.0,
                            value=st.session_state.get('ava_voice_pitch', 1.0),
                            step=0.1,
                            key=f"{key_prefix}voice_pitch_slider",
                            label_visibility="collapsed"
                        )
                        st.session_state.ava_voice_pitch = pitch

                    # Preset buttons
                    st.markdown("**Presets**")
                    preset_cols = st.columns(5)
                    with preset_cols[0]:
                        if st.button("🐢 Slow", use_container_width=True, key=f"{key_prefix}preset_slow"):
                            st.session_state.ava_voice_rate = 0.8
                            st.rerun()
                    with preset_cols[1]:
                        if st.button("🎯 Normal", use_container_width=True, key=f"{key_prefix}preset_normal"):
                            st.session_state.ava_voice_rate = 1.0
                            st.session_state.ava_voice_pitch = 1.0
                            st.rerun()
                    with preset_cols[2]:
                        if st.button("🚀 Fast", use_container_width=True, key=f"{key_prefix}preset_fast"):
                            st.session_state.ava_voice_rate = 1.5
                            st.rerun()
                    with preset_cols[3]:
                        if st.button("🎵 High", use_container_width=True, key=f"{key_prefix}preset_high"):
                            st.session_state.ava_voice_pitch = 1.3
                            st.rerun()
                    with preset_cols[4]:
                        if st.button("📻 Low", use_container_width=True, key=f"{key_prefix}preset_low"):
                            st.session_state.ava_voice_pitch = 0.8
                            st.rerun()

                    # Test voice button
                    if st.button("🎤 Test Voice", use_container_width=True, key=f"{key_prefix}test_voice"):
                        test_text = "Hello! I'm AVA, your trading assistant. This is how I sound with your current settings."
                        WebVoiceHandler.speak_text(test_text)

                    st.markdown("---")
                    st.caption("💡 Voice features use your browser's speech synthesis. Quality may vary by browser.")

            # Model selector moved to top of content column
            st.markdown('</div>', unsafe_allow_html=True)

            # Bring chat input up with negative margin (closer to top)
            st.markdown('<div style="margin-top: -220px;"></div>', unsafe_allow_html=True)

            # Streamlit's native chat input with file attachment support (92% width)
            chat_spacer, chat_col = st.columns([0.08, 0.92])
            with chat_col:
                user_input = st.chat_input(
                    placeholder="How can I help you today?",
                    key=f"{key_prefix}ava_chat_input",
                    accept_file="multiple"  # Enable drag-and-drop file uploads
                )

            # Process message if sent
            if user_input:
                    # Check if user_input is dict (with files) or string (text only)
                    if isinstance(user_input, dict):
                        message_text = str(user_input.get('text', ''))
                        attached_files = user_input.get('files', [])

                        # Build message content with file info
                        content = message_text
                        if attached_files:
                            file_names = [f.name for f in attached_files]
                            content += f"\n\n📎 Attached files: {', '.join(file_names)}"
                    else:
                        # Convert to string to handle ChatInputValue objects
                        message_text = str(user_input)
                        content = str(user_input)
                        attached_files = []

                    # Add user message
                    st.session_state.ava_messages.append({
                        'role': 'user',
                        'content': content,
                        'files': attached_files if attached_files else None
                    })

                    # Get AVA response
                    ava = st.session_state.enhanced_ava
                    response_data = ava.process_message(
                        message_text,
                        user_id=st.session_state.get('user_id', 'web_user'),
                        platform='web'
                    )

                    # Add AVA response with timing and confidence
                    st.session_state.ava_messages.append({
                        'role': 'ava',
                        'content': response_data['response'],
                        'response_time': response_data.get('response_time', 0.0),
                        'confidence': response_data.get('confidence', 0.8)
                    })

                    # Auto-speak if enabled
                    if st.session_state.get('ava_auto_speak', False) and WebVoiceHandler:
                        WebVoiceHandler.speak_text(response_data['response'])

                    # Rerun to show new messages
                    st.rerun()

    # Action buttons below chat input
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)

    with action_col1:
        if st.button("Portfolio", key=f"{key_prefix}ava_portfolio_bottom", use_container_width=True):
            st.session_state.ava_messages.append({'role': 'user', 'content': 'Check my portfolio'})
            ava = st.session_state.enhanced_ava
            response = ava.process_message('Check my portfolio', 'web_user', 'web')
            st.session_state.ava_messages.append({
                'role': 'ava',
                'content': response['response'],
                'response_time': response.get('response_time', 0.0),
                'confidence': response.get('confidence', 0.8)
            })
            st.rerun()

    with action_col2:
        if st.button("Opportunities", key=f"{key_prefix}ava_opportunities_bottom", use_container_width=True):
            st.session_state.ava_messages.append({'role': 'user', 'content': 'Show me opportunities'})
            ava = st.session_state.enhanced_ava
            response = ava.process_message('Show me opportunities', 'web_user', 'web')
            st.session_state.ava_messages.append({
                'role': 'ava',
                'content': response['response'],
                'response_time': response.get('response_time', 0.0),
                'confidence': response.get('confidence', 0.8)
            })
            st.rerun()

    with action_col3:
        if st.button("Watchlist", key=f"{key_prefix}ava_watchlist_bottom", use_container_width=True):
            st.session_state.ava_messages.append({'role': 'user', 'content': 'Analyze my watchlist'})
            ava = st.session_state.enhanced_ava
            response = ava.process_message('Analyze my watchlist', 'web_user', 'web')
            st.session_state.ava_messages.append({
                'role': 'ava',
                'content': response['response'],
                'response_time': response.get('response_time', 0.0),
                'confidence': response.get('confidence', 0.8)
            })
            st.rerun()

    with action_col4:
        if st.button("Help", key=f"{key_prefix}ava_help_bottom", use_container_width=True):
            st.session_state.ava_messages.append({'role': 'user', 'content': 'help'})
            ava = st.session_state.enhanced_ava
            response = ava.process_message('help', 'web_user', 'web')
            st.session_state.ava_messages.append({
                'role': 'ava',
                'content': response['response'],
                'response_time': response.get('response_time', 0.0),
                'confidence': response.get('confidence', 0.8)
            })
            st.rerun()
