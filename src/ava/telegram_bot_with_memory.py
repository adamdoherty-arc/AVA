"""
AVA Telegram Bot with Memory - Enhanced Version
================================================

Enhanced Telegram bot with:
- Persistent user memory across sessions
- Entity tracking (tickers, strategies)
- Conversation summarization
- Local LLM integration (80% cost reduction)
- Personalized responses based on user history

Commands:
/start - Get started with AVA
/portfolio - Portfolio status
/memory - View your stored preferences
/mystocks - Your tracked tickers
/help - Show available commands
"""

import os
import sys
import logging
import tempfile
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment (override any shell env vars with .env file values)
load_dotenv(override=True)

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
except ImportError:
    logger.error("python-telegram-bot not installed. Install: pip install python-telegram-bot")
    sys.exit(1)

try:
    from src.ava.ultimate_ava import get_ultimate_ava
except ImportError:
    logger.error("Ultimate AVA not available")
    sys.exit(1)


class AVATelegramBotWithMemory:
    """AVA Telegram Bot with persistent memory"""

    def __init__(self):
        """Initialize bot with memory-enhanced AVA"""
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token or self.token == "YOUR_BOT_TOKEN_HERE":
            raise ValueError("TELEGRAM_BOT_TOKEN not configured in .env")

        # Initialize Ultimate AVA (includes memory system)
        self.ava = get_ultimate_ava()

        # Track conversation history per user
        self.user_conversations: Dict[int, List[Dict]] = {}

        logger.info("✅ AVA Telegram Bot with Memory initialized")

    def _get_user_id(self, telegram_user_id: int) -> str:
        """Convert Telegram user ID to memory system user ID"""
        return f"telegram:{telegram_user_id}"

    def _get_or_create_conversation(self, telegram_user_id: int) -> List[Dict]:
        """Get or create conversation history for user"""
        if telegram_user_id not in self.user_conversations:
            self.user_conversations[telegram_user_id] = []
        return self.user_conversations[telegram_user_id]

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        chat_id = update.effective_chat.id
        user = update.effective_user
        user_id = self._get_user_id(user.id)

        # Store initial user preference (that they started using AVA)
        if self.ava.memory:
            try:
                from src.ava.memory import MemoryEntry
                memory_entry = MemoryEntry(
                    memory_type="preference",
                    key="telegram_started",
                    value=True,
                    source="telegram_bot"
                )
                self.ava.memory.store_user_memory(user_id, "telegram", memory_entry)
                logger.info(f"Stored initial memory for {user.first_name}")
            except Exception as e:
                logger.error(f"Error storing initial memory: {e}")

        welcome_message = f"""
👋 Hi {user.first_name}!

I'm **AVA** (Autonomous Vector Agent) - your AI trading assistant with **memory**!

🧠 **What Makes Me Special:**
• I **remember** your preferences across sessions
• I **track** your favorite tickers automatically
• I **learn** from our conversations
• I use **local AI** (80% cost reduction!)

💬 **Just Ask Me Anything:**
• "What's a good wheel strategy for AAPL?"
• "How's the market looking today?"
• "What did we discuss last time?"

**New Commands:**
/memory - View your stored preferences
/mystocks - Your tracked tickers
/portfolio - Portfolio status
/help - All commands

**Your Chat ID:** `{chat_id}`

Try asking me about stocks - I'll automatically remember your favorites!
"""
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
        logger.info(f"User {user.first_name} (ID: {chat_id}) started bot")

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
**AVA Commands:**

/start - Get started with AVA
/memory - View your stored preferences
/mystocks - Your tracked tickers
/portfolio - Portfolio status
/status - AVA system status
/help - This help message

**Questions you can ask:**
• "What's a good CSP for TSLA?"
• "How's my portfolio?"
• "What stocks have we discussed?"
• "Explain covered calls"
• "What's the market doing?"

I remember everything we discuss - your preferences, favorite stocks, and strategies!
"""
        await update.message.reply_text(help_text, parse_mode='Markdown')

    async def memory_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /memory command - show user's stored preferences"""
        user = update.effective_user
        user_id = self._get_user_id(user.id)

        if not self.ava.memory:
            await update.message.reply_text("Memory system not available")
            return

        try:
            # Get all user memories
            preferences = self.ava.memory.get_user_memory(user_id, "telegram", "preference")

            if not preferences:
                await update.message.reply_text("No preferences stored yet. Start chatting with me to build your profile!")
                return

            response = "**Your Stored Preferences:**\n\n"

            for pref in preferences[:10]:  # Show top 10
                key = pref['key']
                value = pref['value']
                importance = pref.get('importance', 5)

                response += f"• **{key.replace('_', ' ').title()}**: {value}\n"
                response += f"  _(Importance: {importance}/10)_\n\n"

            # Get memory stats
            stats = self.ava.memory.get_memory_stats(user_id, "telegram")
            response += f"\n**Memory Stats:**\n"
            response += f"• Total memories: {stats.get('total_memories', 0)}\n"
            response += f"• Conversations: {stats.get('conversation_count', 0)}\n"

            await update.message.reply_text(response, parse_mode='Markdown')

        except Exception as e:
            logger.error(f"Error getting memory: {e}")
            await update.message.reply_text(f"Error loading memory: {str(e)}")

    async def mystocks_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /mystocks command - show tracked tickers"""
        user = update.effective_user
        user_id = self._get_user_id(user.id)

        if not self.ava.memory:
            await update.message.reply_text("Memory system not available")
            return

        try:
            # Get tracked tickers
            tracked = self.ava.memory.get_user_entities(
                user_id, "telegram", entity_type="ticker", min_mentions=1
            )

            if not tracked:
                await update.message.reply_text(
                    "No tickers tracked yet! Mention some stock symbols in our conversation and I'll remember them."
                )
                return

            response = "**Your Tracked Stocks:**\n\n"

            for ticker_data in tracked[:15]:  # Show top 15
                ticker = ticker_data['entity_id']
                mentions = ticker_data['mention_count']
                sentiment = ticker_data.get('overall_sentiment', 'neutral')
                interest = ticker_data.get('interest_score', 5)

                # Sentiment emoji
                sentiment_emoji = {
                    'positive': '📈',
                    'negative': '📉',
                    'neutral': '➡️',
                    'mixed': '🔄'
                }.get(sentiment, '➡️')

                response += f"{sentiment_emoji} **${ticker}** - {mentions} mention(s)\n"
                response += f"   Sentiment: {sentiment.title()} | Interest: {interest}/10\n\n"

            await update.message.reply_text(response, parse_mode='Markdown')

        except Exception as e:
            logger.error(f"Error getting tracked stocks: {e}")
            await update.message.reply_text(f"Error loading stocks: {str(e)}")

    async def portfolio_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /portfolio command"""
        try:
            if not self.ava.robinhood_client:
                await update.message.reply_text("Portfolio not connected")
                return

            account = self.ava.robinhood_client.get_account_info()
            positions = self.ava.robinhood_client.get_positions()

            current_value = float(account.get('portfolio_value', 0))
            cash = float(account.get('cash', 0))
            buying_power = float(account.get('buying_power', 0))

            response = f"""
**Portfolio Status:**

💰 Total Value: ${current_value:,.2f}
💵 Cash: ${cash:,.2f}
⚡ Buying Power: ${buying_power:,.2f}
📊 Positions: {len(positions)}
"""
            await update.message.reply_text(response, parse_mode='Markdown')

        except Exception as e:
            logger.error(f"Error getting portfolio: {e}")
            await update.message.reply_text(f"Error loading portfolio: {str(e)}")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            status = self.ava.get_comprehensive_status()
            await update.message.reply_text(f"```\n{status}\n```", parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text(f"Error getting status: {str(e)}")

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages with memory tracking"""
        user = update.effective_user
        user_id = self._get_user_id(user.id)
        user_text = update.message.text

        logger.info(f"Message from {user.first_name}: {user_text}")

        # Get conversation history for this user
        conversation = self._get_or_create_conversation(user.id)

        try:
            # Process message using Ultimate AVA with memory
            response = self.ava.process_message(
                user_id=user_id,
                platform="telegram",
                message=user_text,
                conversation_messages=conversation
            )

            # Send response
            await update.message.reply_text(response)

            logger.info(f"Sent response to {user.first_name} (conversation: {len(conversation)} messages)")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await update.message.reply_text(
                "I apologize, but I encountered an error processing your message. Please try again."
            )

    def run(self):
        """Start the bot"""
        logger.info("🚀 Starting AVA Telegram Bot with Memory...")

        # Create application
        app = Application.builder().token(self.token).build()

        # Add handlers
        app.add_handler(CommandHandler("start", self.start_command))
        app.add_handler(CommandHandler("help", self.help_command))
        app.add_handler(CommandHandler("memory", self.memory_command))
        app.add_handler(CommandHandler("mystocks", self.mystocks_command))
        app.add_handler(CommandHandler("portfolio", self.portfolio_command))
        app.add_handler(CommandHandler("status", self.status_command))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))

        # Start polling
        logger.info("✅ AVA Telegram Bot with Memory is running!")
        logger.info("Features: Persistent memory, entity tracking, local LLM, 80% cost reduction!")
        app.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    """Main entry point"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     🧠 AVA Telegram Bot with MEMORY - Starting... 🧠             ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

Enhanced Features:
✅ Persistent user memory across sessions
✅ Automatic ticker tracking
✅ Conversation summarization (70-90% token reduction)
✅ Local LLM integration (80% cost reduction)
✅ Personalized responses based on history

AVA is connecting to Telegram...

Once connected:
1. Open Telegram
2. Find your bot
3. Send /start to begin
4. Try: "What's a good CSP for TSLA?"

I'll automatically remember your preferences and favorite stocks!

Press Ctrl+C to stop the bot
""")

    try:
        bot = AVATelegramBotWithMemory()
        bot.run()
    except KeyboardInterrupt:
        print("\n\n👋 AVA Telegram Bot stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
