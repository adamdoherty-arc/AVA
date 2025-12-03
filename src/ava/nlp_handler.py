"""
AVA Natural Language Understanding Handler
==========================================

Uses FREE LLM service (Groq/Gemini/DeepSeek) to understand natural language
queries and route them to appropriate AVA command handlers.

Features:
- Intent detection from natural language
- Entity extraction (tickers, dates, etc.)
- Command routing
- Conversational responses
- Zero cost (uses existing FREE LLM infrastructure)

Author: Claude Code
Date: 2025-11-10
"""

import logging
from typing import Dict, Optional, List
from enum import Enum

# Import existing FREE LLM service
from src.services.llm_service import LLMService
from src.rag import RAGService

logger = logging.getLogger(__name__)


class Intent(Enum):
    """Supported intents for AVA commands"""
    PORTFOLIO = "portfolio"
    POSITIONS = "positions"
    OPPORTUNITIES = "opportunities"
    TRADINGVIEW = "tradingview"
    XTRADES = "xtrades"
    TASKS = "tasks"
    STATUS = "status"
    HELP = "help"
    UNKNOWN = "unknown"


class NaturalLanguageHandler:
    """
    Natural language understanding for AVA Telegram bot

    Uses existing FREE LLM service to understand user queries and route to commands
    """

    def __init__(self) -> None:
        self.rag = RAGService(collection_name='magnus_knowledge')
        """Initialize NLP handler with FREE LLM service"""
        self.llm_service = LLMService()  # Auto-selects: groq > deepseek > gemini (FREE!)
        logger.info("✅ NLP Handler initialized with FREE LLM service")

    def parse_intent(self, user_text: str, context: Optional[Dict] = None) -> Dict:
        """
        Parse user's natural language query to detect intent and entities

        Args:
            user_text: User's message text
            context: Optional conversation context (previous intent, entities)

        Returns:
            Dictionary with intent, entities, and suggested response
        """
        # Build intent detection prompt with context
        prompt = self._build_intent_prompt(user_text, context)

        try:
            # Use FREE LLM service (Groq/Gemini/DeepSeek - $0.00 cost!)
            response = self.llm_service.generate(
                prompt=prompt,
                provider=None,  # Auto-selects best FREE provider
                max_tokens=300,
                temperature=0.1,  # Low temp for consistent intent detection
                use_cache=True  # Cache similar queries
            )

            # Parse LLM response
            result = self._parse_llm_response(response['text'])
            result['model_used'] = f"{response['provider']}/{response['model']}"
            result['cost'] = response['cost']
            result['original_query'] = user_text

            logger.info(
                f"Intent detected: {result['intent']} "
                f"(model: {result['model_used']}, cost: ${result['cost']:.4f})"
            )

            return result

        except Exception as e:
            logger.error(f"Error parsing intent with LLM: {e}")
            logger.info("Falling back to keyword-based intent detection")
            # Fallback to simple keyword matching
            return self.fallback_intent_detection(user_text)

    def _build_intent_prompt(self, user_text: str, context: Optional[Dict] = None) -> str:
        """Build prompt for intent detection with optional context"""
        context_info = ""
        if context:
            prev_intent = context.get('previous_intent')
            prev_entities = context.get('previous_entities', {})
            if prev_intent:
                context_info = f"\n**Previous Context:**\n- User previously asked about: {prev_intent}\n"
                if prev_entities.get('tickers'):
                    context_info += f"- Mentioned tickers: {', '.join(prev_entities['tickers'])}\n"

        return f"""You are an intent classifier for AVA, a trading assistant Telegram bot.
{context_info}
Analyze this user query and determine the intent:

**User Query:** "{user_text}"

**Available Intents:**
1. PORTFOLIO - User wants portfolio balance, performance, or overview
   Examples: "How's my portfolio?", "What's my balance?", "Portfolio status"

2. POSITIONS - User wants to see their active options positions
   Examples: "Show my positions", "What trades do I have?", "My options"

3. OPPORTUNITIES - User wants CSP trading opportunities
   Examples: "What are good trades?", "CSP opportunities", "Best plays"

4. TRADINGVIEW - User wants TradingView watchlists or charts
   Examples: "TradingView alerts", "Show me charts", "Watchlists"

5. XTRADES - User wants Xtrades following/signals
   Examples: "Xtrades alerts", "Who am I following?", "Trader signals"

6. TASKS - User wants to know what AVA is working on
   Examples: "What tasks are running?", "What are you doing?", "AVA status"

7. STATUS - User wants system status/health
   Examples: "Are you online?", "System status", "Bot health"

8. HELP - User needs help or doesn't know what to ask
   Examples: "Help", "What can you do?", "Commands"

9. UNKNOWN - Query doesn't match any intent

**Your Task:**
Respond with ONLY these 4 lines (no extra text):

INTENT: [intent name from list above]
CONFIDENCE: [0.0-1.0 confidence score]
ENTITIES: [comma-separated list of tickers/symbols if mentioned, or "none"]
RESPONSE_HINT: [Brief suggestion for what to say, like "Show portfolio with balance and daily change"]

Example 1:
User: "How are my trades doing?"
Response:
INTENT: POSITIONS
CONFIDENCE: 0.9
ENTITIES: none
RESPONSE_HINT: Show active options positions with P&L

Example 2:
User: "What's the best play on NVDA?"
Response:
INTENT: OPPORTUNITIES
CONFIDENCE: 0.85
ENTITIES: NVDA
RESPONSE_HINT: Show CSP opportunities filtered for NVDA

Now classify this query:
User: "{user_text}"

Your response (4 lines only):"""

    def _parse_llm_response(self, llm_text: str) -> Dict:
        """Parse LLM response into structured format"""
        result = {
            'intent': Intent.UNKNOWN.value,
            'confidence': 0.0,
            'entities': {},
            'response_hint': "Try /help to see available commands"
        }

        try:
            lines = llm_text.strip().split('\n')

            for line in lines:
                line = line.strip()

                if line.startswith('INTENT:'):
                    intent_str = line.replace('INTENT:', '').strip().upper()
                    # Try to match to Intent enum
                    for intent in Intent:
                        if intent.name == intent_str:
                            result['intent'] = intent.value
                            break

                elif line.startswith('CONFIDENCE:'):
                    conf_str = line.replace('CONFIDENCE:', '').strip()
                    try:
                        result['confidence'] = float(conf_str)
                    except ValueError:
                        result['confidence'] = 0.5

                elif line.startswith('ENTITIES:'):
                    entities_str = line.replace('ENTITIES:', '').strip()
                    if entities_str.lower() != 'none':
                        # Extract tickers
                        tickers = [t.strip().upper() for t in entities_str.split(',') if t.strip()]
                        if tickers:
                            result['entities']['tickers'] = tickers

                elif line.startswith('RESPONSE_HINT:'):
                    result['response_hint'] = line.replace('RESPONSE_HINT:', '').strip()

        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")

        return result

    def get_command_for_intent(self, intent: str) -> Optional[str]:
        """
        Map intent to command method name

        Args:
            intent: Intent string (e.g., 'portfolio', 'positions')

        Returns:
            Command method name (e.g., 'portfolio_command') or None
        """
        intent_to_command = {
            Intent.PORTFOLIO.value: 'portfolio_command',
            Intent.POSITIONS.value: 'positions_command',
            Intent.OPPORTUNITIES.value: 'opportunities_command',
            Intent.TRADINGVIEW.value: 'tradingview_command',
            Intent.XTRADES.value: 'xtrades_command',
            Intent.TASKS.value: 'tasks_command',
            Intent.STATUS.value: 'status_command',
            Intent.HELP.value: 'help_command',
        }

        return intent_to_command.get(intent)

    def should_use_nlp(self, user_text: str) -> bool:
        """
        Determine if text should be processed with NLP or handled differently

        Args:
            user_text: User's message

        Returns:
            True if should use NLP, False otherwise
        """
        # Don't use NLP for very short messages
        if len(user_text.strip()) < 3:
            return False

        # Don't use NLP if it looks like a command (starts with /)
        if user_text.strip().startswith('/'):
            return False

        return True

    def fallback_intent_detection(self, user_text: str) -> Dict:
        """
        Simple keyword-based fallback when LLM is unavailable

        Args:
            user_text: User's message text

        Returns:
            Dictionary with intent based on keyword matching
        """
        user_text_lower = user_text.lower()

        # Keyword patterns for each intent
        patterns = {
            Intent.PORTFOLIO: ['portfolio', 'balance', 'account', 'money', 'capital', 'net worth'],
            Intent.POSITIONS: ['position', 'trade', 'option', 'holding', 'contracts', 'what do i have'],
            Intent.OPPORTUNITIES: ['opportun', 'play', 'trade idea', 'good', 'best', 'csp', 'sell put'],
            Intent.TRADINGVIEW: ['tradingview', 'chart', 'watchlist', 'tv'],
            Intent.XTRADES: ['xtrades', 'follow', 'trader', 'signal', 'alert'],
            Intent.TASKS: ['task', 'working', 'doing', 'busy', 'activity'],
            Intent.STATUS: ['status', 'online', 'health', 'running', 'alive'],
            Intent.HELP: ['help', 'command', 'what can', 'show me', 'how to'],
        }

        # Check each pattern
        for intent, keywords in patterns.items():
            for keyword in keywords:
                if keyword in user_text_lower:
                    return {
                        'intent': intent.value,
                        'confidence': 0.6,  # Medium confidence for fallback
                        'entities': {},
                        'response_hint': f'Using keyword fallback for {intent.value}',
                        'model_used': 'keyword_fallback',
                        'cost': 0.0,
                        'original_query': user_text
                    }

        # Unknown intent fallback
        return {
            'intent': Intent.UNKNOWN.value,
            'confidence': 0.3,
            'entities': {},
            'response_hint': 'Try using /help to see available commands',
            'model_used': 'keyword_fallback',
            'cost': 0.0,
            'original_query': user_text
        }


# Quick test function
def test_nlp_handler():
    """Test the NLP handler with sample queries"""
    print("=== Testing AVA NLP Handler ===\n")

    handler = NaturalLanguageHandler()

    test_queries = [
        "How's my portfolio?",
        "What positions do I have?",
        "Show me the best CSP opportunities",
        "What are you working on?",
        "Are you online?",
        "What can you do?",
        "NVDA analysis",
        "How are my trades doing?"
    ]

    for query in test_queries:
        print(f"Query: \"{query}\"")
        result = handler.parse_intent(query)
        print(f"  Intent: {result['intent']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Entities: {result.get('entities', {})}")
        print(f"  Hint: {result['response_hint']}")
        print(f"  Model: {result['model_used']}")
        print(f"  Cost: ${result['cost']:.4f}")
        print()


if __name__ == "__main__":
    test_nlp_handler()


    def query_knowledge_base(self, query: str) -> dict:
        """
        Query the RAG knowledge base for information

        Args:
            query: User's question

        Returns:
            Dictionary with answer, confidence, and sources
        """
        try:
            result = self.rag.query(query, use_cache=True)

            return {
                "success": True,
                "answer": result.answer,
                "confidence": result.confidence,
                "sources": [
                    {
                        "file": s.metadata.get("file_path", "unknown"),
                        "relevance": s.score
                    }
                    for s in result.sources[:3]
                ]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
