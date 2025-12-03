"""
Telegram Adapter for AVA
Provides Telegram bot integration using AVA Core
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from src.ava.core import AVACore, MessageResponse
import os

logger = logging.getLogger(__name__)


class TelegramAVAAdapter:
    """Telegram adapter for AVA Core"""

    def __init__(self, ava_core: Optional[AVACore] = None, bot_token: Optional[str] = None):
        """
        Initialize Telegram adapter

        Args:
            ava_core: AVA Core instance (creates new if None)
            bot_token: Telegram bot token (from env if None)
        """
        self.ava = ava_core or AVACore()
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN must be set")

        # Initialize bot application
        self.application = Application.builder().token(self.bot_token).build()
        
        # Register handlers
        self._register_handlers()
        
        logger.info("TelegramAVAAdapter initialized")

    def _register_handlers(self) -> None:
        """Register Telegram bot handlers"""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CommandHandler("help", self._help_command))
        
        # Message handler (text messages)
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
        
        # Callback query handler (for inline keyboards)
        self.application.add_handler(CallbackQueryHandler(self._handle_callback))

    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_id = str(update.effective_user.id)
        
        welcome_message = (
            "👋 Hello! I'm AVA, your expert AI trading assistant.\n\n"
            "I can help you with:\n"
            "• Portfolio analysis\n"
            "• Trading opportunities\n"
            "• Watchlist analysis\n"
            "• Task management\n"
            "• And much more!\n\n"
            "Just send me a message or use /help to see commands."
        )
        
        # Quick action buttons
        keyboard = [
            [
                InlineKeyboardButton("📊 Portfolio", callback_data="quick_portfolio"),
                InlineKeyboardButton("📈 Opportunities", callback_data="quick_opportunities")
            ],
            [
                InlineKeyboardButton("📝 Watchlist", callback_data="quick_watchlist"),
                InlineKeyboardButton("❓ Help", callback_data="quick_help")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(welcome_message, reply_markup=reply_markup)

    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = (
            "🤖 AVA Commands:\n\n"
            "/start - Start conversation\n"
            "/help - Show this help\n\n"
            "You can also just send me messages like:\n"
            "• 'What's my portfolio balance?'\n"
            "• 'Analyze NVDA watchlist'\n"
            "• 'Show me trading opportunities'\n"
            "• 'What tasks are pending?'\n\n"
            "I'll understand and help you!"
        )
        await update.message.reply_text(help_text)

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages"""
        user_id = str(update.effective_user.id)
        message_text = update.message.text
        
        # Show typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        # Process message with AVA
        response_chunks = []
        async for chunk in self.ava.process_message(
            message=message_text,
            user_id=user_id,
            platform="telegram"
        ):
            response_chunks.append(chunk)
        
        full_response = "".join(response_chunks)
        
        # Send response
        await update.message.reply_text(full_response)

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline keyboard callbacks"""
        query = update.callback_query
        await query.answer()
        
        user_id = str(update.effective_user.id)
        callback_data = query.data
        
        # Map callback data to messages
        callback_map = {
            "quick_portfolio": "What's my portfolio status?",
            "quick_opportunities": "Show me the best trading opportunities",
            "quick_watchlist": "Analyze my default watchlist",
            "quick_help": "What can you help me with?"
        }
        
        message = callback_map.get(callback_data, "Help")
        
        # Show typing indicator
        await context.bot.send_chat_action(chat_id=query.message.chat_id, action="typing")
        
        # Process message
        response_chunks = []
        async for chunk in self.ava.process_message(
            message=message,
            user_id=user_id,
            platform="telegram"
        ):
            response_chunks.append(chunk)
        
        full_response = "".join(response_chunks)
        
        # Edit message with response
        await query.edit_message_text(full_response)

    def start(self) -> None:
        """Start the Telegram bot"""
        logger.info("Starting Telegram bot...")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)

    async def stop(self) -> None:
        """Stop the Telegram bot"""
        logger.info("Stopping Telegram bot...")
        await self.application.stop()
        await self.application.shutdown()


# Standalone bot runner
def run_telegram_bot():
    """Run Telegram bot standalone"""
    adapter = TelegramAVAAdapter()
    try:
        adapter.start()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Error running bot: {e}")


if __name__ == "__main__":
    run_telegram_bot()

