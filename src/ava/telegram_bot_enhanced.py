"""
AVA Telegram Bot - Enhanced Production Version
===============================================

Production-ready Telegram bot with comprehensive features:
- User authentication and authorization
- Rate limiting
- Rich inline keyboards
- Portfolio and position tracking
- Chart generation
- TradingView and Xtrades integration
- Error handling and logging

Commands:
/start - Get started with AVA
/portfolio - Portfolio status with interactive buttons
/positions - Options positions with actions
/opportunities - CSP opportunities
/tradingview - TradingView watchlists and alerts
/xtrades - Xtrades followed traders
/tasks - AVA task status
/help - Show available commands
/status - System status
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List
from functools import wraps
from dotenv import load_dotenv

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment
load_dotenv(override=True)

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Telegram imports
try:
    from telegram import Update
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        CallbackQueryHandler,
        filters,
        ContextTypes
    )
except ImportError:
    logger.error("python-telegram-bot not installed. Install: pip install python-telegram-bot")
    sys.exit(1)

# AVA imports
try:
    from src.ava.voice_handler import AVAVoiceHandler
    from src.ava.db_manager import get_db_manager, DatabaseError
    from src.ava.rate_limiter import get_rate_limiter, RateLimitExceeded
    from src.ava.magnus_integration import MagnusIntegration
    from src.ava.charts import ChartGenerator
    from src.ava.nlp_handler import NaturalLanguageHandler
    from src.ava import inline_keyboards as kb
except ImportError as e:
    logger.error(f"AVA modules not available: {e}")
    AVAVoiceHandler = None


class AVATelegramBot:
    """Enhanced AVA Telegram Bot with full feature set"""

    def __init__(self) -> None:
        """Initialize bot"""
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token or self.token == "YOUR_BOT_TOKEN_HERE":
            raise ValueError("TELEGRAM_BOT_TOKEN not configured in .env")

        # Load authorized users
        authorized_users_str = os.getenv("TELEGRAM_AUTHORIZED_USERS", "")
        self.authorized_users = set()
        if authorized_users_str:
            try:
                self.authorized_users = set(int(uid.strip()) for uid in authorized_users_str.split(",") if uid.strip())
                logger.info(f"Loaded {len(self.authorized_users)} authorized users")
            except ValueError as e:
                logger.error(f"Error parsing TELEGRAM_AUTHORIZED_USERS: {e}")

        # Initialize components
        self.voice_handler = AVAVoiceHandler() if AVAVoiceHandler else None
        self.rate_limiter = get_rate_limiter()
        self.magnus = MagnusIntegration()
        self.charts = ChartGenerator()
        self.nlp_handler = NaturalLanguageHandler()

        # Conversation context tracking (user_id -> context)
        self.conversation_contexts = {}

        logger.info("✅ AVA Telegram Bot initialized with NLP and context tracking")

    # ==================== Authentication Decorator ====================

    def require_auth(self) -> None:
        """Decorator to require authentication for commands"""
        def decorator(func):
            @wraps(func)
            async def wrapper(self_or_update, *args, **kwargs):
                # Handle both methods and standalone functions
                if isinstance(self_or_update, Update):
                    update = self_or_update
                    self_instance = args[0] if args else self
                else:
                    self_instance = self_or_update
                    update = args[0] if args else kwargs.get('update')

                user_id = update.effective_user.id
                username = update.effective_user.username or "Unknown"

                # Check if user is authorized
                if self_instance.authorized_users and user_id not in self_instance.authorized_users:
                    logger.warning(f"Unauthorized access attempt by {username} (ID: {user_id})")
                    await update.message.reply_text(
                        "🚫 Unauthorized Access\n\n"
                        f"Your user ID: `{user_id}`\n\n"
                        "Please contact the administrator to authorize your account.\n"
                        "Add your user ID to TELEGRAM_AUTHORIZED_USERS in .env",
                        parse_mode='Markdown'
                    )
                    return

                # Call the original function
                if isinstance(self_or_update, Update):
                    return await func(update, *args, **kwargs)
                else:
                    return await func(self_or_update, *args, **kwargs)

            return wrapper
        return decorator

    def with_rate_limit(self) -> None:
        """Decorator to apply rate limiting"""
        def decorator(func):
            @wraps(func)
            async def wrapper(self_or_update, *args, **kwargs):
                if isinstance(self_or_update, Update):
                    update = self_or_update
                    self_instance = args[0] if args else self
                else:
                    self_instance = self_or_update
                    update = args[0] if args else kwargs.get('update')

                user_id = update.effective_user.id

                try:
                    self_instance.rate_limiter.check_rate_limit(user_id)
                except RateLimitExceeded as e:
                    await update.message.reply_text(str(e))
                    return

                if isinstance(self_or_update, Update):
                    return await func(update, *args, **kwargs)
                else:
                    return await func(self_or_update, *args, **kwargs)

            return wrapper
        return decorator

    # ==================== Command Handlers ====================

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        chat_id = update.effective_chat.id
        user = update.effective_user
        user_id = user.id

        # Check if authorized
        is_authorized = not self.authorized_users or user_id in self.authorized_users

        welcome_message = f"""
👋 Hi {user.first_name}!

I'm **AVA** (Automated Vector Agent) - your AI trading assistant for 24/7 portfolio management!

🎤 **Voice:** Send me a voice message and I'll transcribe and respond
💬 **Text:** Or just type your questions
📊 **Interactive:** Use buttons for quick actions

**Your Chat ID:** `{chat_id}`
**Your User ID:** `{user_id}`
**Status:** {'✅ Authorized' if is_authorized else '🚫 Unauthorized'}

{'' if is_authorized else '**⚠️ Please contact administrator to authorize your account.**'}

**What I can help with:**
• Portfolio updates and performance
• Options positions and Greeks
• CSP opportunities analysis
• TradingView watchlists and alerts
• Xtrades trader following
• Stock charts and analysis
• Task tracking and status

**Commands:**
/portfolio - Your portfolio with charts
/positions - Options positions
/opportunities - Best CSP plays
/tradingview - TradingView integration
/xtrades - Xtrades following
/tasks - What I'm working on
/status - System status
/help - Detailed help

**💬 Natural Language:** Just ask me questions!
• "How's my portfolio?"
• "What positions do I have?"
• "Show me the best opportunities"
• "Are there any good trades?"
• "What are you working on?"

I'll understand and respond appropriately! 🧠
"""
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
        logger.info(f"User {user.first_name} (ID: {user_id}) started bot [Authorized: {is_authorized}]")

    async def portfolio_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /portfolio command"""
        # Auth check
        user_id = update.effective_user.id
        if self.authorized_users and user_id not in self.authorized_users:
            await update.message.reply_text("🚫 Unauthorized")
            return

        # Rate limit check
        try:
            self.rate_limiter.check_rate_limit(user_id)
        except RateLimitExceeded as e:
            await update.message.reply_text(str(e))
            return
        try:
            # Get portfolio data
            portfolio = await self.magnus.get_portfolio_summary()

            if portfolio.get('balance') is None:
                await update.message.reply_text(
                    "📊 No portfolio data available.\n"
                    "Please sync your Robinhood account first."
                )
                return

            # Format message
            balance = portfolio['balance']
            daily_change = portfolio['daily_change']
            daily_change_pct = portfolio['daily_change_pct']
            timestamp = portfolio['timestamp']

            change_emoji = "📈" if daily_change >= 0 else "📉"
            change_sign = "+" if daily_change >= 0 else ""

            message = f"""
💼 **Portfolio Summary**

**Balance:** ${balance:,.2f}
{change_emoji} **Today:** {change_sign}${daily_change:,.2f} ({change_sign}{daily_change_pct:.2f}%)

**Last Updated:** {timestamp.strftime('%I:%M %p')}
"""

            if portfolio.get('notes'):
                message += f"\n📝 {portfolio['notes']}"

            # Send with inline keyboard
            await update.message.reply_text(
                message,
                parse_mode='Markdown',
                reply_markup=kb.build_portfolio_keyboard()
            )

        except DatabaseError as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")
        except Exception as e:
            logger.error(f"Error in portfolio command: {e}", exc_info=True)
            await update.message.reply_text("❌ An unexpected error occurred. Please try again.")

    async def positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command"""
        # Auth check
        user_id = update.effective_user.id
        if self.authorized_users and user_id not in self.authorized_users:
            await update.message.reply_text("🚫 Unauthorized")
            return

        # Rate limit check
        try:
            self.rate_limiter.check_rate_limit(user_id)
        except RateLimitExceeded as e:
            await update.message.reply_text(str(e))
            return
        try:
            positions = await self.magnus.get_options_positions()

            if not positions:
                await update.message.reply_text(
                    "📊 No active positions found.",
                    reply_markup=kb.build_positions_keyboard()
                )
                return

            # Format positions
            message = "📊 **Active Options Positions**\n\n"

            for pos in positions[:10]:  # Show max 10
                ticker = pos['ticker']
                option_type = pos['option_type'].upper()
                strike = pos['strike']
                exp = pos['expiration_date'].strftime('%m/%d')
                quantity = pos['quantity']
                pnl = pos.get('unrealized_pnl', 0)
                pnl_emoji = "🟢" if pnl >= 0 else "🔴"

                message += f"{pnl_emoji} **{ticker}** {quantity}x ${strike}{option_type[0]} exp {exp}\n"
                message += f"   P&L: ${pnl:,.2f}\n\n"

            if len(positions) > 10:
                message += f"\n_...and {len(positions) - 10} more positions_"

            await update.message.reply_text(
                message,
                parse_mode='Markdown',
                reply_markup=kb.build_positions_keyboard()
            )

        except DatabaseError as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")
        except Exception as e:
            logger.error(f"Error in positions command: {e}", exc_info=True)
            await update.message.reply_text("❌ An unexpected error occurred.")

    async def opportunities_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE, filter_tickers: Optional[List[str]] = None):
        """Handle /opportunities command with optional ticker filtering"""
        # Auth check
        user_id = update.effective_user.id
        if self.authorized_users and user_id not in self.authorized_users:
            await update.message.reply_text("🚫 Unauthorized")
            return

        # Rate limit check
        try:
            self.rate_limiter.check_rate_limit(user_id)
        except RateLimitExceeded as e:
            await update.message.reply_text(str(e))
            return
        try:
            opps = await self.magnus.get_csp_opportunities(limit=10)

            # Filter by tickers if provided
            if filter_tickers:
                opps = [opp for opp in opps if opp['ticker'].upper() in [t.upper() for t in filter_tickers]]
                if not opps:
                    await update.message.reply_text(
                        f"🎯 No CSP opportunities found for: {', '.join(filter_tickers)}\n"
                        "Try checking other tickers or ask without filtering."
                    )
                    return

            if not opps:
                await update.message.reply_text("🎯 No CSP opportunities found.")
                return

            message = "🎯 **Top CSP Opportunities**\n\n"

            for opp in opps[:5]:
                ticker = opp['ticker']
                strike = opp['strike_price']
                premium = opp['premium']
                exp = opp['expiration_date'].strftime('%m/%d')
                annual_return = opp.get('annual_return', 0)

                message += f"**{ticker}** ${strike} exp {exp}\n"
                message += f"💰 Premium: ${premium:.2f} | Return: {annual_return:.1f}%\n\n"

            await update.message.reply_text(
                message,
                parse_mode='Markdown',
                reply_markup=kb.build_csp_opportunities_keyboard(opps)
            )

        except DatabaseError as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")
        except Exception as e:
            logger.error(f"Error in opportunities command: {e}", exc_info=True)
            await update.message.reply_text("❌ An unexpected error occurred.")

    async def tradingview_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /tradingview command"""
        # Auth check
        user_id = update.effective_user.id
        if self.authorized_users and user_id not in self.authorized_users:
            await update.message.reply_text("🚫 Unauthorized")
            return

        # Rate limit check
        try:
            self.rate_limiter.check_rate_limit(user_id)
        except RateLimitExceeded as e:
            await update.message.reply_text(str(e))
            return
        await update.message.reply_text(
            "📺 **TradingView Integration**\n\n"
            "Access your TradingView watchlists, alerts, and charts.",
            parse_mode='Markdown',
            reply_markup=kb.build_tradingview_keyboard()
        )

    async def xtrades_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /xtrades command"""
        # Auth check
        user_id = update.effective_user.id
        if self.authorized_users and user_id not in self.authorized_users:
            await update.message.reply_text("🚫 Unauthorized")
            return

        # Rate limit check
        try:
            self.rate_limiter.check_rate_limit(user_id)
        except RateLimitExceeded as e:
            await update.message.reply_text(str(e))
            return
        try:
            traders = await self.magnus.get_following_traders()

            message = f"👥 **Xtrades Following** ({len(traders)})\n\n"

            if traders:
                for trader in traders[:5]:
                    username = trader['username']
                    alerts = "🔔" if trader['alerts_enabled'] else "🔕"
                    message += f"{alerts} @{username}\n"

                if len(traders) > 5:
                    message += f"\n_...and {len(traders) - 5} more traders_"
            else:
                message += "Not following any traders yet."

            await update.message.reply_text(
                message,
                parse_mode='Markdown',
                reply_markup=kb.build_xtrades_keyboard()
            )

        except DatabaseError as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")
        except Exception as e:
            logger.error(f"Error in xtrades command: {e}", exc_info=True)
            await update.message.reply_text("❌ An unexpected error occurred.")

    async def tasks_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /tasks command"""
        # Auth check
        user_id = update.effective_user.id
        if self.authorized_users and user_id not in self.authorized_users:
            await update.message.reply_text("🚫 Unauthorized")
            return

        # Rate limit check
        try:
            self.rate_limiter.check_rate_limit(user_id)
        except RateLimitExceeded as e:
            await update.message.reply_text(str(e))
            return
        try:
            stats = await self.magnus.get_task_stats()
            active_tasks = await self.magnus.get_active_tasks()

            message = f"""
📋 **AVA Task Status**

**Queue:**
• In Progress: {stats['in_progress']}
• Proposed: {stats['proposed']}
• Completed: {stats['completed']}

**Recent Activity:**
• Completed today: {stats['completed_today']}
• This week: {stats['completed_this_week']}
"""

            if active_tasks:
                message += "\n**Current Tasks:**\n"
                for task in active_tasks[:3]:
                    status_emoji = "🔄" if task['status'] == 'in_progress' else "📋"
                    message += f"{status_emoji} {task['title']}\n"

            await update.message.reply_text(
                message,
                parse_mode='Markdown',
                reply_markup=kb.build_tasks_keyboard()
            )

        except DatabaseError as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")
        except Exception as e:
            logger.error(f"Error in tasks command: {e}", exc_info=True)
            await update.message.reply_text("❌ An unexpected error occurred.")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            db = get_db_manager()
            db_stats = db.get_stats()

            status_text = f"""
🤖 **AVA System Status**

**Core Systems:**
• Database: {'✅ Connected' if db_stats['initialized'] else '❌ Disconnected'}
• Voice Handler: {'✅ Online' if self.voice_handler else '❌ Offline'}
• Rate Limiter: ✅ Active
• Magnus Integration: ✅ Active

**Connection Pool:**
• Min connections: {db_stats['min_connections']}
• Max connections: {db_stats['max_connections']}

**Bot Status:**
• Telegram: ✅ Active
• Authorized users: {len(self.authorized_users) if self.authorized_users else 'All'}

AVA is monitoring your portfolio 24/7!
"""
            await update.message.reply_text(status_text, parse_mode='Markdown')

        except Exception as e:
            logger.error(f"Error in status command: {e}")
            await update.message.reply_text("❌ Error getting system status")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
📚 **AVA Help & Commands**

**💬 Natural Language (NEW!):**
You can now ask me questions in plain English! Just type naturally:
• "How's my portfolio doing?"
• "What positions do I have?"
• "Show me the best opportunities"
• "Are there any good trades?"
• "What are you working on?"
• "Are you online?"

I'll understand your intent and respond accordingly! 🧠

**📋 Direct Commands:**
If you prefer commands, you can still use:
/portfolio - View portfolio balance and performance
/positions - View active options positions
/opportunities - Find best CSP opportunities
/tradingview - TradingView watchlists and alerts
/xtrades - Xtrades followed traders and signals
/tasks - View AVA's current tasks
/status - System status and health
/help - This help message

**🎤 Voice Messages:**
Send me a voice message and I'll transcribe and respond!

**🔘 Interactive Buttons:**
Use the inline buttons on messages for quick actions.

**💡 Tips:**
• Ask questions naturally - I'll figure out what you need
• Use specific tickers: "What's NVDA doing?"
• Check multiple things: "Portfolio and positions"
"""
        await update.message.reply_text(
            help_text,
            parse_mode='Markdown',
            reply_markup=kb.build_help_keyboard()
        )

    # ==================== Callback Handlers ====================

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard callbacks"""
        query = update.callback_query
        await query.answer()

        callback_data = query.data
        user_id = query.from_user.id

        # Rate limiting
        try:
            self.rate_limiter.check_rate_limit(user_id)
        except RateLimitExceeded as e:
            await query.edit_message_text(str(e))
            return

        # Route to appropriate handler
        if callback_data.startswith("portfolio_"):
            await self.handle_portfolio_callback(query, callback_data)
        elif callback_data.startswith("positions_") or callback_data.startswith("pos_"):
            await self.handle_positions_callback(query, callback_data)
        elif callback_data.startswith("show_"):
            await self.handle_navigation_callback(query, callback_data)
        else:
            await query.edit_message_text("This feature is coming soon!")

    async def handle_portfolio_callback(self, query, callback_data: str):
        """Handle portfolio-related callbacks"""
        if callback_data == "portfolio_refresh":
            # Refresh portfolio data
            portfolio = await self.magnus.get_portfolio_summary()
            # ... (format and send update)
            await query.edit_message_text("🔄 Portfolio refreshed!")

        elif callback_data == "portfolio_chart":
            # Generate and send chart
            await query.edit_message_text("📊 Generating chart...")
            try:
                balances = await self.magnus.get_balance_history(days=30)
                chart_path = self.charts.generate_portfolio_chart(balances)

                with open(chart_path, 'rb') as photo:
                    await query.message.reply_photo(photo=photo, caption="📈 Portfolio Balance - Last 30 Days")

                await query.delete_message()
            except Exception as e:
                logger.error(f"Error generating chart: {e}")
                await query.edit_message_text("❌ Error generating chart")

    async def handle_positions_callback(self, query, callback_data: str):
        """Handle position-related callbacks"""
        await query.edit_message_text(f"Processing: {callback_data}")

    async def handle_navigation_callback(self, query, callback_data: str):
        """Handle navigation callbacks"""
        command_map = {
            "show_portfolio": self.portfolio_command,
            "show_positions": self.positions_command,
            "show_opportunities": self.opportunities_command,
            "show_tradingview": self.tradingview_command,
            "show_xtrades": self.xtrades_command,
            "show_tasks": self.tasks_command,
        }

        if callback_data in command_map:
            # Create fake update to call command
            await query.edit_message_text("🔄 Loading...")

    # ==================== Message Handlers ====================

    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle voice messages"""
        # Check auth and rate limit
        user_id = update.effective_user.id
        if self.authorized_users and user_id not in self.authorized_users:
            await update.message.reply_text("🚫 Unauthorized")
            return

        try:
            self.rate_limiter.check_rate_limit(user_id)
        except RateLimitExceeded as e:
            await update.message.reply_text(str(e))
            return

        if not self.voice_handler:
            await update.message.reply_text("🎤 Voice processing not available yet")
            return

        await update.message.reply_text("🎤 Transcribing your voice message...")

        try:
            # Download and process voice
            voice_file = await update.message.voice.get_file()
            voice_path = f"temp_voice_{update.message.message_id}.ogg"
            await voice_file.download_to_drive(voice_path)

            transcribed_text = self.voice_handler.transcribe_voice(voice_path)

            if not transcribed_text:
                await update.message.reply_text("Sorry, couldn't transcribe your message")
                return

            await update.message.reply_text(f"📝 *You said:* {transcribed_text}", parse_mode='Markdown')

            # Process query
            response = self.voice_handler.process_query(transcribed_text)
            await update.message.reply_text(response['response_text'])

            # Cleanup
            if os.path.exists(voice_path):
                os.remove(voice_path)

        except Exception as e:
            logger.error(f"Error handling voice: {e}")
            await update.message.reply_text("❌ Error processing voice message")

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages with natural language understanding"""
        user_id = update.effective_user.id
        user_text = update.message.text
        username = update.effective_user.first_name or "User"

        # Check auth
        if self.authorized_users and user_id not in self.authorized_users:
            await update.message.reply_text("🚫 Unauthorized")
            return

        # Rate limiting
        try:
            self.rate_limiter.check_rate_limit(user_id)
        except RateLimitExceeded as e:
            await update.message.reply_text(str(e))
            return

        # Skip NLP for very short or command-like messages
        if not self.nlp_handler.should_use_nlp(user_text):
            await update.message.reply_text(
                "I didn't quite understand that. Try:\n"
                "• Asking a question like 'How's my portfolio?'\n"
                "• Using /help to see available commands"
            )
            return

        # Show typing indicator
        await update.message.chat.send_action("typing")

        logger.info(f"User {username} (ID: {user_id}) asked: \"{user_text}\"")

        # Get conversation context for this user
        context = self.conversation_contexts.get(user_id)

        # Parse intent using NLP with context
        try:
            intent_result = self.nlp_handler.parse_intent(user_text, context=context)
            intent = intent_result['intent']
            confidence = intent_result['confidence']
            entities = intent_result.get('entities', {})

            logger.info(
                f"Detected intent: {intent} "
                f"(confidence: {confidence:.2f}, "
                f"model: {intent_result['model_used']}, "
                f"cost: ${intent_result['cost']:.4f})"
            )

            # Low confidence - ask for clarification
            if confidence < 0.5:
                await update.message.reply_text(
                    f"I'm not quite sure what you're asking for. {intent_result['response_hint']}\n\n"
                    "Try using /help to see what I can do!"
                )
                return

            # Get command method for intent
            command_method_name = self.nlp_handler.get_command_for_intent(intent)

            if not command_method_name:
                await update.message.reply_text(
                    "I understood your question, but I don't have that feature yet!\n"
                    "Use /help to see what I can do."
                )
                return

            # Get the command method
            command_method = getattr(self, command_method_name, None)

            if not command_method:
                await update.message.reply_text(
                    "Something went wrong routing your request. Try using /help"
                )
                return

            # Provide conversational acknowledgment
            acknowledgments = {
                'portfolio': f"📊 Sure {username}, let me get your portfolio...",
                'positions': f"📋 Checking your positions {username}...",
                'opportunities': f"🎯 Finding the best opportunities for you...",
                'tradingview': f"📺 Accessing TradingView...",
                'xtrades': f"👥 Checking Xtrades...",
                'tasks': f"📋 Checking my task list...",
                'status': f"🤖 Getting system status...",
                'help': f"📚 Here's what I can do..."
            }

            ack_message = acknowledgments.get(intent, "Processing your request...")

            # Add ticker info to acknowledgment if present
            if entities.get('tickers'):
                tickers_str = ', '.join(entities['tickers'])
                ack_message += f"\n💡 Filtering for: {tickers_str}"

            await update.message.reply_text(ack_message)

            # Call the appropriate command with entity filtering
            if intent == 'opportunities' and entities.get('tickers'):
                # Pass tickers to opportunities command for filtering
                await command_method(update, context, filter_tickers=entities['tickers'])
            else:
                await command_method(update, context)

            # Update conversation context for this user
            self.conversation_contexts[user_id] = {
                'previous_intent': intent,
                'previous_entities': entities,
                'previous_query': user_text,
                'timestamp': update.message.date
            }

            # Log successful processing
            logger.info(f"Successfully processed {intent} for user {username} (context updated)")

        except Exception as e:
            logger.error(f"Error handling text message: {e}", exc_info=True)
            await update.message.reply_text(
                "❌ Sorry, I encountered an error processing your request.\n"
                "Try using /help or a specific command like /portfolio"
            )

    # ==================== Run ====================

    def run(self) -> None:
        """Start the bot"""
        logger.info("🚀 Starting AVA Telegram Bot...")

        # Create application
        app = Application.builder().token(self.token).build()

        # Add handlers
        app.add_handler(CommandHandler("start", self.start_command))
        app.add_handler(CommandHandler("help", self.help_command))
        app.add_handler(CommandHandler("portfolio", self.portfolio_command))
        app.add_handler(CommandHandler("positions", self.positions_command))
        app.add_handler(CommandHandler("opportunities", self.opportunities_command))
        app.add_handler(CommandHandler("tradingview", self.tradingview_command))
        app.add_handler(CommandHandler("xtrades", self.xtrades_command))
        app.add_handler(CommandHandler("tasks", self.tasks_command))
        app.add_handler(CommandHandler("status", self.status_command))
        app.add_handler(CallbackQueryHandler(self.handle_callback))
        app.add_handler(MessageHandler(filters.VOICE, self.handle_voice))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))

        # Start polling
        logger.info("✅ AVA Telegram Bot is running!")
        logger.info(f"Authorized users: {len(self.authorized_users) if self.authorized_users else 'All'}")
        app.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    """Main entry point"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║       🤖 AVA Telegram Bot - Enhanced Production Version 🤖       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

AVA is connecting to Telegram...

Features:
✅ User authentication
✅ Rate limiting
✅ Interactive keyboards
✅ Portfolio tracking
✅ Chart generation
✅ TradingView integration
✅ Xtrades integration

Press Ctrl+C to stop the bot
""")

    try:
        bot = AVATelegramBot()
        bot.run()
    except KeyboardInterrupt:
        print("\n\n👋 AVA Telegram Bot stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
