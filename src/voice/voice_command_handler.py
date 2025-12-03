"""
Voice Command Handler - Wake Word Detection & Command Processing
================================================================

Enables hands-free interaction with AVA through voice commands.

Features:
- Wake word detection ("Hey AVA", "Magnus", "Computer")
- Speech-to-text processing
- Command pattern matching
- Multi-command workflows
- Context-aware command interpretation
- Integration with AVA agent system

Author: Magnus Trading Platform
Created: 2025-11-21
"""

import logging
import re
import os
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


# ============================================================================
# Wake Word Configuration
# ============================================================================

class WakeWord(Enum):
    """Available wake words for voice activation"""
    HEY_AVA = "hey_ava"
    MAGNUS = "magnus"
    COMPUTER = "computer"
    ASSISTANT = "assistant"


@dataclass
class WakeWordConfig:
    """Configuration for wake word detection"""
    wake_word: WakeWord
    sensitivity: float = 0.5  # 0.0 (lenient) to 1.0 (strict)
    timeout_seconds: int = 5  # How long to listen after wake word
    confirmation_sound: bool = True
    visual_feedback: bool = True


# ============================================================================
# Command Patterns
# ============================================================================

class CommandCategory(Enum):
    """Categories of voice commands"""
    PORTFOLIO = "portfolio"
    MARKET_DATA = "market_data"
    ANALYSIS = "analysis"
    TRADING = "trading"
    NAVIGATION = "navigation"
    SETTINGS = "settings"
    HELP = "help"


@dataclass
class CommandPattern:
    """Voice command pattern definition"""
    category: CommandCategory
    patterns: List[str]  # Regex patterns to match
    handler: str  # Handler function name
    description: str
    examples: List[str]
    requires_parameter: bool = False
    parameter_type: Optional[str] = None  # "ticker", "number", "date", etc.


# ============================================================================
# Command Registry
# ============================================================================

COMMAND_PATTERNS = [
    # Portfolio Commands
    CommandPattern(
        category=CommandCategory.PORTFOLIO,
        patterns=[
            r"show (my )?portfolio",
            r"what('s| is) my (current )?balance",
            r"(show|display) (my )?positions",
            r"how (am i|did i) do(ing)? today",
        ],
        handler="show_portfolio",
        description="Display portfolio overview and positions",
        examples=["Show my portfolio", "What's my balance", "Display positions"]
    ),

    # Market Data Commands
    CommandPattern(
        category=CommandCategory.MARKET_DATA,
        patterns=[
            r"(what('s| is)|get|show) (the )?(price|quote) (of |for )?(.+)",
            r"how('s| is) (.+) (doing|performing|trading)",
            r"(show|get) (.+) (chart|graph)",
        ],
        handler="get_stock_quote",
        description="Get current stock price and chart",
        examples=["What's the price of Apple", "How's Tesla doing", "Show NVDA chart"],
        requires_parameter=True,
        parameter_type="ticker"
    ),

    CommandPattern(
        category=CommandCategory.MARKET_DATA,
        patterns=[
            r"(show|display|what are) (the )?(market|futures|indices)",
            r"how('s| is) (the )?market (doing|performing)",
        ],
        handler="show_market_overview",
        description="Display market indices and futures",
        examples=["Show market", "How's the market doing", "Display futures"]
    ),

    # Analysis Commands
    CommandPattern(
        category=CommandCategory.ANALYSIS,
        patterns=[
            r"analyze (.+)",
            r"(what|should) (do|should) i (do |trade )(with |on )?(.+)",
            r"(give me|show) (.+) (analysis|breakdown|report)",
        ],
        handler="analyze_stock",
        description="Perform comprehensive stock analysis",
        examples=["Analyze Apple", "What should I do with Tesla", "Show AMD analysis"],
        requires_parameter=True,
        parameter_type="ticker"
    ),

    CommandPattern(
        category=CommandCategory.ANALYSIS,
        patterns=[
            r"(find|show|scan for) (options|option) (opportunities|plays|trades)",
            r"(what are|show me) (the )?(best|top) (options|trades)",
            r"scan (for )?(cash secured puts|covered calls|spreads)",
        ],
        handler="scan_options",
        description="Scan for options trading opportunities",
        examples=["Find options opportunities", "Show best trades", "Scan for cash secured puts"]
    ),

    # Trading Commands
    CommandPattern(
        category=CommandCategory.TRADING,
        patterns=[
            r"buy (\d+) (shares of )?(.+)",
            r"sell (\d+) (shares of )?(.+)",
            r"(place|create|open) (a )?(.+) (trade|order|position)",
        ],
        handler="execute_trade",
        description="Execute a trade (confirmation required)",
        examples=["Buy 10 shares of Apple", "Sell 5 shares of Tesla"],
        requires_parameter=True,
        parameter_type="trade_params"
    ),

    # Navigation Commands
    CommandPattern(
        category=CommandCategory.NAVIGATION,
        patterns=[
            r"(go to|open|show|navigate to) (the )?(.+) (page|screen|tab)",
            r"(switch to|change to) (.+)",
        ],
        handler="navigate_to_page",
        description="Navigate to different pages",
        examples=["Go to dashboard", "Open options page", "Switch to positions"],
        requires_parameter=True,
        parameter_type="page_name"
    ),

    # Settings Commands
    CommandPattern(
        category=CommandCategory.SETTINGS,
        patterns=[
            r"(change|switch|set) (my )?personality to (.+)",
            r"(be more|act more|speak more) (.+)",
            r"(use|change to) (.+) (mode|personality|style)",
        ],
        handler="change_personality",
        description="Change AVA's personality mode",
        examples=["Change personality to analyst", "Be more friendly", "Use professional mode"],
        requires_parameter=True,
        parameter_type="personality"
    ),

    CommandPattern(
        category=CommandCategory.SETTINGS,
        patterns=[
            r"(enable|turn on|activate) (.+)",
            r"(disable|turn off|deactivate) (.+)",
        ],
        handler="toggle_setting",
        description="Enable or disable settings",
        examples=["Enable voice feedback", "Turn on notifications", "Disable alerts"],
        requires_parameter=True,
        parameter_type="setting_name"
    ),

    # Help Commands
    CommandPattern(
        category=CommandCategory.HELP,
        patterns=[
            r"(what can you do|help|commands|capabilities)",
            r"(how do i|teach me how to) (.+)",
            r"(show|list) (all )?(available )?commands",
        ],
        handler="show_help",
        description="Display available commands and help",
        examples=["What can you do", "Help", "Show commands"]
    ),
]


# ============================================================================
# Voice Command Handler
# ============================================================================

class VoiceCommandHandler:
    """
    Main voice command processing system

    Handles wake word detection, speech-to-text, command matching,
    and integration with AVA's agent system.
    """

    def __init__(self, wake_word_config: Optional[WakeWordConfig] = None):
        """
        Initialize voice command handler

        Args:
            wake_word_config: Wake word configuration (defaults to HEY_AVA)
        """
        self.config = wake_word_config or WakeWordConfig(wake_word=WakeWord.HEY_AVA)
        self.listening = False
        self.command_history: List[Dict[str, Any]] = []

        # Command handlers registry
        self.handlers: Dict[str, Callable] = {}
        self._register_handlers()

        logger.info(f"Voice command handler initialized with wake word: {self.config.wake_word.value}")

    def _register_handlers(self) -> None:
        """Register all command handlers"""
        self.handlers = {
            "show_portfolio": self._handle_show_portfolio,
            "get_stock_quote": self._handle_get_stock_quote,
            "show_market_overview": self._handle_show_market_overview,
            "analyze_stock": self._handle_analyze_stock,
            "scan_options": self._handle_scan_options,
            "execute_trade": self._handle_execute_trade,
            "navigate_to_page": self._handle_navigate_to_page,
            "change_personality": self._handle_change_personality,
            "toggle_setting": self._handle_toggle_setting,
            "show_help": self._handle_show_help,
        }

    def process_voice_input(self, transcript: str) -> Dict[str, Any]:
        """
        Process voice input and execute matched command

        Args:
            transcript: Speech-to-text transcript

        Returns:
            Dict with status, command, response, and action
        """
        transcript_lower = transcript.lower().strip()

        # Check for wake word if not already listening
        if not self.listening:
            if self._detect_wake_word(transcript_lower):
                self.listening = True
                return {
                    "status": "wake_word_detected",
                    "message": "I'm listening. What would you like me to do?",
                    "listening": True
                }
            else:
                return {
                    "status": "idle",
                    "message": "Say the wake word to activate voice commands",
                    "listening": False
                }

        # Match command pattern
        matched_command = self._match_command(transcript_lower)

        if not matched_command:
            self.listening = False
            return {
                "status": "no_match",
                "message": "I didn't understand that command. Say 'help' for available commands.",
                "listening": False,
                "transcript": transcript
            }

        # Execute command
        try:
            handler = self.handlers.get(matched_command["handler"])
            if not handler:
                raise ValueError(f"Handler not found: {matched_command['handler']}")

            result = handler(matched_command)

            # Log to history
            self.command_history.append({
                "timestamp": pd.Timestamp.now(),
                "transcript": transcript,
                "command": matched_command["pattern"].category.value,
                "handler": matched_command["handler"],
                "success": result.get("status") == "success"
            })

            self.listening = False
            return result

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            self.listening = False
            return {
                "status": "error",
                "message": f"Sorry, I encountered an error: {str(e)}",
                "listening": False
            }

    def _detect_wake_word(self, text: str) -> bool:
        """Detect wake word in text"""
        wake_words = {
            WakeWord.HEY_AVA: ["hey ava", "hi ava", "hello ava"],
            WakeWord.MAGNUS: ["magnus", "hey magnus", "hi magnus"],
            WakeWord.COMPUTER: ["computer", "hey computer"],
            WakeWord.ASSISTANT: ["assistant", "hey assistant"]
        }

        patterns = wake_words.get(self.config.wake_word, [])
        return any(pattern in text for pattern in patterns)

    def _match_command(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Match text against command patterns

        Returns:
            Dict with matched pattern, handler, and extracted parameters
        """
        for pattern in COMMAND_PATTERNS:
            for regex in pattern.patterns:
                match = re.search(regex, text, re.IGNORECASE)
                if match:
                    # Extract parameters from regex groups
                    parameters = {}
                    if pattern.requires_parameter:
                        # Get the last captured group as parameter
                        groups = [g for g in match.groups() if g is not None]
                        if groups:
                            parameters["value"] = groups[-1].strip()
                            parameters["type"] = pattern.parameter_type

                    return {
                        "pattern": pattern,
                        "handler": pattern.handler,
                        "parameters": parameters,
                        "matched_text": match.group(0)
                    }

        return None

    # ========================================================================
    # Command Handlers
    # ========================================================================

    def _handle_show_portfolio(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle portfolio display command"""
        return {
            "status": "success",
            "action": "show_portfolio",
            "message": "Displaying your portfolio",
            "page": "dashboard",
            "section": "portfolio"
        }

    def _handle_get_stock_quote(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stock quote request"""
        ticker = command["parameters"].get("value", "")

        # Clean ticker (remove common words)
        ticker = re.sub(r'\b(stock|of|for|the)\b', '', ticker, flags=re.IGNORECASE).strip()
        ticker = ticker.upper()

        return {
            "status": "success",
            "action": "get_quote",
            "message": f"Getting quote for {ticker}",
            "ticker": ticker,
            "page": "market_data"
        }

    def _handle_show_market_overview(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle market overview request"""
        return {
            "status": "success",
            "action": "show_market",
            "message": "Displaying market overview",
            "page": "dashboard",
            "section": "market_indices"
        }

    def _handle_analyze_stock(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stock analysis request"""
        ticker = command["parameters"].get("value", "")
        ticker = ticker.upper()

        return {
            "status": "success",
            "action": "analyze_stock",
            "message": f"Analyzing {ticker}",
            "ticker": ticker,
            "page": "options_analysis",
            "analysis_type": "comprehensive"
        }

    def _handle_scan_options(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle options scanning request"""
        return {
            "status": "success",
            "action": "scan_options",
            "message": "Scanning for options opportunities",
            "page": "options_trading_hub",
            "tab": "screening"
        }

    def _handle_execute_trade(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle trade execution request (requires confirmation)"""
        trade_params = command["parameters"].get("value", "")

        return {
            "status": "confirmation_required",
            "action": "execute_trade",
            "message": f"Trade request: {trade_params}. Please confirm verbally or tap to confirm.",
            "trade_params": trade_params,
            "requires_confirmation": True
        }

    def _handle_navigate_to_page(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle page navigation"""
        page_name = command["parameters"].get("value", "")

        # Map common page names
        page_mapping = {
            "dashboard": "dashboard",
            "home": "dashboard",
            "portfolio": "dashboard",
            "options": "options_trading_hub",
            "trades": "options_trading_hub",
            "positions": "positions_page_improved",
            "sports": "sports_betting_hub_page",
            "betting": "sports_betting_hub_page",
            "monitoring": "system_monitoring_hub_page",
            "settings": "system_management_hub_page",
            "calendar": "calendar_spreads_page",
            "earnings": "earnings_calendar_page",
        }

        page = page_mapping.get(page_name.lower(), page_name)

        return {
            "status": "success",
            "action": "navigate",
            "message": f"Navigating to {page_name}",
            "page": page
        }

    def _handle_change_personality(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle personality change request"""
        personality = command["parameters"].get("value", "")

        # Map personality names
        personality_mapping = {
            "professional": "PROFESSIONAL",
            "friendly": "FRIENDLY",
            "witty": "WITTY",
            "mentor": "MENTOR",
            "concise": "CONCISE",
            "charming": "CHARMING",
            "analyst": "ANALYST",
            "coach": "COACH",
            "rebel": "REBEL",
            "guru": "GURU"
        }

        mode = personality_mapping.get(personality.lower(), personality.upper())

        return {
            "status": "success",
            "action": "change_personality",
            "message": f"Switching to {personality} mode",
            "personality": mode
        }

    def _handle_toggle_setting(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle settings toggle"""
        setting = command["parameters"].get("value", "")

        # Determine enable/disable from matched text
        enable = "enable" in command["matched_text"].lower() or "turn on" in command["matched_text"].lower()

        return {
            "status": "success",
            "action": "toggle_setting",
            "message": f"{'Enabling' if enable else 'Disabling'} {setting}",
            "setting": setting,
            "enable": enable
        }

    def _handle_show_help(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Handle help request"""
        # Group commands by category
        commands_by_category = {}
        for pattern in COMMAND_PATTERNS:
            category = pattern.category.value
            if category not in commands_by_category:
                commands_by_category[category] = []
            commands_by_category[category].append({
                "description": pattern.description,
                "examples": pattern.examples[:2]  # First 2 examples
            })

        return {
            "status": "success",
            "action": "show_help",
            "message": "Here are the available voice commands",
            "commands": commands_by_category
        }

    def get_command_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent command history"""
        return self.command_history[-limit:]

    def clear_history(self) -> None:
        """Clear command history"""
        self.command_history.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get voice command usage statistics"""
        if not self.command_history:
            return {
                "total_commands": 0,
                "success_rate": 0,
                "most_used_command": None
            }

        total = len(self.command_history)
        successful = sum(1 for cmd in self.command_history if cmd["success"])

        # Count command usage
        command_counts = {}
        for cmd in self.command_history:
            command_counts[cmd["command"]] = command_counts.get(cmd["command"], 0) + 1

        most_used = max(command_counts.items(), key=lambda x: x[1])[0] if command_counts else None

        return {
            "total_commands": total,
            "success_rate": round((successful / total) * 100, 1),
            "most_used_command": most_used,
            "commands_by_category": command_counts
        }


# ============================================================================
# Wake Word Detection (Picovoice Porcupine Integration)
# ============================================================================

class WakeWordDetector:
    """
    Wake word detection using Picovoice Porcupine

    Supports offline wake word detection with custom wake words.
    Requires Picovoice access key (free tier available).
    """

    def __init__(self, wake_word: WakeWord = WakeWord.HEY_AVA):
        """
        Initialize wake word detector

        Args:
            wake_word: Wake word to detect
        """
        self.wake_word = wake_word
        self.porcupine = None
        self.audio_stream = None
        self.is_listening = False

        # Check if Picovoice is available
        try:
            import pvporcupine
            self._initialize_porcupine()
        except ImportError:
            logger.warning("Picovoice Porcupine not installed. Wake word detection disabled.")
            logger.info("Install with: pip install pvporcupine")

    def _initialize_porcupine(self) -> None:
        """Initialize Porcupine wake word engine"""
        try:
            import pvporcupine

            access_key = os.getenv('PICOVOICE_ACCESS_KEY')
            if not access_key:
                logger.warning("PICOVOICE_ACCESS_KEY not set. Wake word detection disabled.")
                return

            # Built-in wake words: "porcupine", "bumblebee", "computer", "jarvis", etc.
            # For custom wake words, train at https://console.picovoice.ai/

            keyword = self._map_wake_word_to_keyword()

            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=[keyword]
            )

            logger.info(f"Wake word detector initialized: {keyword}")

        except Exception as e:
            logger.error(f"Failed to initialize wake word detector: {e}")

    def _map_wake_word_to_keyword(self) -> str:
        """Map WakeWord enum to Porcupine keyword"""
        mapping = {
            WakeWord.HEY_AVA: "jarvis",  # Use "jarvis" as closest match
            WakeWord.MAGNUS: "computer",
            WakeWord.COMPUTER: "computer",
            WakeWord.ASSISTANT: "jarvis"
        }
        return mapping.get(self.wake_word, "computer")

    def start_listening(self, callback: Callable[[bool], None]):
        """
        Start listening for wake word

        Args:
            callback: Function to call when wake word detected
        """
        if not self.porcupine:
            logger.warning("Wake word detector not initialized")
            return

        import pyaudio
        import struct

        self.is_listening = True

        pa = pyaudio.PyAudio()

        audio_stream = pa.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length
        )

        try:
            while self.is_listening:
                pcm = audio_stream.read(self.porcupine.frame_length)
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)

                keyword_index = self.porcupine.process(pcm)

                if keyword_index >= 0:
                    logger.info(f"Wake word detected: {self.wake_word.value}")
                    callback(True)

        finally:
            audio_stream.close()
            pa.terminate()

    def stop_listening(self) -> None:
        """Stop listening for wake word"""
        self.is_listening = False
        if self.porcupine:
            self.porcupine.delete()


# ============================================================================
# Helper Functions
# ============================================================================

def get_available_commands() -> List[Dict[str, Any]]:
    """Get list of all available voice commands"""
    commands = []
    for pattern in COMMAND_PATTERNS:
        commands.append({
            "category": pattern.category.value,
            "description": pattern.description,
            "examples": pattern.examples,
            "requires_parameter": pattern.requires_parameter
        })
    return commands


def format_help_text() -> str:
    """Format help text for display"""
    commands_by_category = {}
    for pattern in COMMAND_PATTERNS:
        category = pattern.category.value.replace("_", " ").title()
        if category not in commands_by_category:
            commands_by_category[category] = []
        commands_by_category[category].append({
            "description": pattern.description,
            "examples": pattern.examples
        })

    help_text = "**Available Voice Commands:**\n\n"
    for category, commands in commands_by_category.items():
        help_text += f"**{category}**\n"
        for cmd in commands:
            help_text += f"- {cmd['description']}\n"
            for example in cmd['examples'][:2]:
                help_text += f"  - *\"{example}\"*\n"
        help_text += "\n"

    return help_text


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    import pandas as pd

    # Test voice command handler
    handler = VoiceCommandHandler()

    test_commands = [
        "Hey AVA",
        "Show my portfolio",
        "What's the price of Apple",
        "Analyze Tesla",
        "Go to options page",
        "Change personality to analyst",
        "Help"
    ]

    print("Testing Voice Commands\n" + "="*50)
    for cmd in test_commands:
        print(f"\nInput: {cmd}")
        result = handler.process_voice_input(cmd)
        print(f"Status: {result['status']}")
        print(f"Message: {result.get('message', 'N/A')}")
        if result.get('action'):
            print(f"Action: {result['action']}")

    # Show statistics
    print("\n" + "="*50)
    print("Statistics:")
    stats = handler.get_statistics()
    print(json.dumps(stats, indent=2))
