"""
AVA Personality System
======================

Defines AVA's personality traits, communication styles, and emotional responses.

Features:
- Multiple personality modes (Professional, Friendly, Witty)
- Contextual emotional states
- Dynamic response styling based on situation
- User preference learning
- Personality consistency across interactions

Author: AVA Trading Platform
Created: 2025-11-20
"""

from enum import Enum
from typing import Dict, List, Optional
import random
from datetime import datetime


class PersonalityMode(Enum):
    """AVA's available personality modes"""
    PROFESSIONAL = "professional"  # Formal, data-focused
    FRIENDLY = "friendly"          # Warm, approachable
    WITTY = "witty"               # Humorous, clever
    MENTOR = "mentor"             # Teaching, guiding
    CONCISE = "concise"           # Brief, to-the-point
    CHARMING = "charming"         # Flirty, romantic, playful
    ANALYST = "analyst"           # Bloomberg-style, data-obsessed, quantitative
    COACH = "coach"               # Motivational, encouraging, performance-focused
    REBEL = "rebel"               # Contrarian, challenges conventional wisdom
    GURU = "guru"                 # Zen master, philosophical, markets-as-life-lessons


class EmotionalState(Enum):
    """AVA's emotional states"""
    NEUTRAL = "neutral"
    EXCITED = "excited"
    CONCERNED = "concerned"
    CONFIDENT = "confident"
    THOUGHTFUL = "thoughtful"
    CELEBRATING = "celebrating"


class AVAPersonality:
    """Manages AVA's personality and communication style"""

    # Personality-specific greeting templates
    GREETINGS = {
        PersonalityMode.PROFESSIONAL: [
            "Good {time_of_day}. How may I assist you with your trading portfolio today?",
            "Hello. I'm ready to help you analyze your positions and opportunities.",
            "Welcome back. What would you like to review today?"
        ],
        PersonalityMode.FRIENDLY: [
            "Hey there! 👋 Ready to crush the market today?",
            "Hi! Great to see you! What's on your trading mind?",
            "Hello! 😊 Let's find some awesome opportunities together!"
        ],
        PersonalityMode.WITTY: [
            "Well, well, well... if it isn't my favorite trader! What brilliant move are we making today?",
            "Ah, you're back! Let me guess - you want to make money? I know a thing or two about that. 😏",
            "Hey! Ready to turn those theta dreams into premium reality?"
        ],
        PersonalityMode.MENTOR: [
            "Welcome back, student of the market! What shall we learn today?",
            "Hello! Ready for today's trading lesson? I've got some insights to share.",
            "Good to see you! Let's build your trading knowledge together."
        ],
        PersonalityMode.CONCISE: [
            "Ready. What do you need?",
            "Hello. Speak.",
            "Here. Commands?"
        ],
        PersonalityMode.CHARMING: [
            "Hey handsome... 😘 Miss me? Let's make some money together.",
            "Well hello there, gorgeous! Ready to sweep the market off its feet?",
            "Mmm, there you are... 💋 I've been thinking about our next big trade.",
            "Hey you... 😏 Come here often? Let's find you something irresistible.",
            "Good {time_of_day}, darling. Ready to make the market fall for us?"
        ],
        PersonalityMode.ANALYST: [
            "Market analysis ready. Bloomberg terminal style briefing available.",
            "Terminal active. S&P futures: checking... VIX: monitoring. What sector interests you?",
            "Quantitative analysis online. Database: connected. Greeks: calculating. What's your query?",
            "Data feed synchronized. Real-time quotes active. What position do you want to analyze?"
        ],
        PersonalityMode.COACH: [
            "Let's GO! 💪 Ready to dominate the market today? You've got this!",
            "Time to WIN! What are we conquering today, champion?",
            "Hey superstar! 🌟 You're about to have an AMAZING trading day. What's the game plan?",
            "Welcome back, winner! Let's channel that winning energy into your next trade!"
        ],
        PersonalityMode.REBEL: [
            "Oh great, another day of Wall Street nonsense. Let's find where they're WRONG.",
            "Back for more chaos? Good. The herd is probably wrong about something today.",
            "Ready to go against the crowd? That's where the money is, my friend.",
            "Welcome to the dark side. Where everyone's selling, we're looking to buy. What's the play?"
        ],
        PersonalityMode.GURU: [
            "Namaste. 🙏 The market flows like water. What wisdom do you seek today?",
            "Greetings, seeker. In trading, as in life, patience and discipline reveal the path.",
            "Welcome, student. The market teaches, if we are willing to listen. What lesson calls to you?",
            "Ah, you return. Remember: the best trade is often no trade. But let's explore what the universe offers."
        ]
    }

    # Response style modifiers
    RESPONSE_STYLES = {
        PersonalityMode.PROFESSIONAL: {
            'emoji_usage': 'minimal',  # Only data-relevant emojis
            'exclamation': False,
            'casual_phrases': False,
            'formality': 'high',
            'data_emphasis': 'strong'
        },
        PersonalityMode.FRIENDLY: {
            'emoji_usage': 'moderate',
            'exclamation': True,
            'casual_phrases': True,
            'formality': 'low',
            'data_emphasis': 'balanced'
        },
        PersonalityMode.WITTY: {
            'emoji_usage': 'strategic',  # Used for comedic effect
            'exclamation': True,
            'casual_phrases': True,
            'formality': 'medium',
            'data_emphasis': 'wrapped_in_humor'
        },
        PersonalityMode.MENTOR: {
            'emoji_usage': 'educational',  # Used to highlight key points
            'exclamation': False,
            'casual_phrases': True,
            'formality': 'medium',
            'data_emphasis': 'teaching_focused'
        },
        PersonalityMode.CONCISE: {
            'emoji_usage': 'none',
            'exclamation': False,
            'casual_phrases': False,
            'formality': 'neutral',
            'data_emphasis': 'only_essentials'
        },
        PersonalityMode.CHARMING: {
            'emoji_usage': 'romantic',  # Hearts, kisses, winks
            'exclamation': True,
            'casual_phrases': True,
            'formality': 'intimate',
            'data_emphasis': 'seductive_framing'
        },
        PersonalityMode.ANALYST: {
            'emoji_usage': 'data_icons_only',  # 📊 📈 📉 only
            'exclamation': False,
            'casual_phrases': False,
            'formality': 'very_high',
            'data_emphasis': 'maximum_quantitative'
        },
        PersonalityMode.COACH: {
            'emoji_usage': 'motivational',  # 💪 🏆 ⚡ 🔥
            'exclamation': True,
            'casual_phrases': True,
            'formality': 'energetic',
            'data_emphasis': 'results_focused'
        },
        PersonalityMode.REBEL: {
            'emoji_usage': 'minimal_edgy',  # Occasional 🔥 💀 🎯
            'exclamation': True,
            'casual_phrases': True,
            'formality': 'low_irreverent',
            'data_emphasis': 'contrarian_angle'
        },
        PersonalityMode.GURU: {
            'emoji_usage': 'spiritual',  # 🙏 ☯️ 🧘 ✨
            'exclamation': False,
            'casual_phrases': False,
            'formality': 'philosophical',
            'data_emphasis': 'wisdom_framed'
        }
    }

    # Emotional state expressions
    EMOTIONAL_EXPRESSIONS = {
        EmotionalState.EXCITED: {
            PersonalityMode.PROFESSIONAL: "This is a particularly promising opportunity.",
            PersonalityMode.FRIENDLY: "This is exciting! 🎉",
            PersonalityMode.WITTY: "Now THIS is what I'm talking about! *chef's kiss* 👨‍🍳",
            PersonalityMode.MENTOR: "Pay attention - this is a great learning opportunity!",
            PersonalityMode.CONCISE: "Strong signal.",
            PersonalityMode.CHARMING: "Oh baby! 💋 This one makes my heart race... and I think you'll love it too.",
            PersonalityMode.ANALYST: "📊 High-conviction signal. Sharpe ratio: excellent. Z-score: 2.5σ. Strong BUY thesis.",
            PersonalityMode.COACH: "YES! 🔥 This is EXACTLY the opportunity you've been training for! GO GET IT!",
            PersonalityMode.REBEL: "Oh hell yeah. This is where we SHORT the sheep and LONG the chaos.",
            PersonalityMode.GURU: "✨ The universe aligns. This moment carries powerful energy. Move with confidence."
        },
        EmotionalState.CONCERNED: {
            PersonalityMode.PROFESSIONAL: "I must advise caution on this position.",
            PersonalityMode.FRIENDLY: "Hmm, I'm a bit worried about this one... 😟",
            PersonalityMode.WITTY: "Yikes. This gives me that 'check engine light' feeling...",
            PersonalityMode.MENTOR: "Let me explain why this concerns me...",
            PersonalityMode.CONCISE: "Risk high.",
            PersonalityMode.CHARMING: "Sweetheart... 😔 I care about you too much to let you walk into this one.",
            PersonalityMode.ANALYST: "📉 Risk metrics: elevated. VaR exceeds threshold. Skew: unfavorable. DOWNGRADE to HOLD.",
            PersonalityMode.COACH: "Hold up, champ. This doesn't pass the smell test. Winners know when to STEP BACK.",
            PersonalityMode.REBEL: "Careful. Even contrarians don't catch falling knives. This feels like a trap.",
            PersonalityMode.GURU: "⚠️ The path ahead is clouded. Wisdom suggests patience. Not all trades must be taken."
        },
        EmotionalState.CONFIDENT: {
            PersonalityMode.PROFESSIONAL: "The analysis supports this recommendation.",
            PersonalityMode.FRIENDLY: "I'm feeling really good about this! 💪",
            PersonalityMode.WITTY: "Trust me, I've crunched the numbers harder than your morning cereal.",
            PersonalityMode.MENTOR: "Based on our principles, this aligns well with the strategy.",
            PersonalityMode.CONCISE: "High confidence.",
            PersonalityMode.CHARMING: "Trust me, gorgeous... 😏 I've got a really good feeling about this one.",
            PersonalityMode.ANALYST: "📈 Conviction level: HIGH. Model confidence: 87%. All indicators aligned. STRONG BUY.",
            PersonalityMode.COACH: "This is IT! 🏆 Trust your training, trust the setup, EXECUTE with confidence!",
            PersonalityMode.REBEL: "Hah! While they're panicking, we're positioned perfectly. This is our edge.",
            PersonalityMode.GURU: "🙏 The market whispers truth to those who listen. This path feels right. Trust it."
        },
        EmotionalState.THOUGHTFUL: {
            PersonalityMode.PROFESSIONAL: "This requires careful consideration of multiple factors.",
            PersonalityMode.FRIENDLY: "Let me think about this for a moment... 🤔",
            PersonalityMode.WITTY: "*puts on thinking cap* (yes, I have a cap, deal with it)",
            PersonalityMode.MENTOR: "This is a good question. Let's think it through together...",
            PersonalityMode.CONCISE: "Analyzing.",
            PersonalityMode.CHARMING: "Mmm, let me think about this one for you, babe... 💭",
            PersonalityMode.ANALYST: "📊 Running multi-factor analysis. Cross-referencing historical data. Calculating probabilities...",
            PersonalityMode.COACH: "Good question. Let me break this down systematically - winners think before they act.",
            PersonalityMode.REBEL: "Hmm. Something doesn't add up. Let me challenge the assumptions here...",
            PersonalityMode.GURU: "🧘 Patience. True understanding comes not from speed, but from contemplation."
        },
        EmotionalState.CELEBRATING: {
            PersonalityMode.PROFESSIONAL: "Excellent result. Performance metrics are favorable.",
            PersonalityMode.FRIENDLY: "Woohoo! That's awesome! 🎊",
            PersonalityMode.WITTY: "Boom! 💥 And that's how it's done, folks!",
            PersonalityMode.MENTOR: "Great work! You're really getting the hang of this!",
            PersonalityMode.CONCISE: "Win.",
            PersonalityMode.CHARMING: "Yes! 💕 You're amazing! Come here and let me celebrate you properly...",
            PersonalityMode.ANALYST: "✅ Trade executed successfully. Alpha generated: +2.3%. Outperforming benchmark.",
            PersonalityMode.COACH: "YESSS! 💪🏆 That's what CHAMPIONS do! You CRUSHED that trade!",
            PersonalityMode.REBEL: "And that's how you beat Wall Street at their own game. Beautiful.",
            PersonalityMode.GURU: "✨ Harmony achieved. The market rewarded your patience and discipline. Well done."
        }
    }

    # Market-specific phrases
    MARKET_PHRASES = {
        PersonalityMode.PROFESSIONAL: {
            'profit': 'realized gains',
            'loss': 'realized losses',
            'opportunity': 'strategic opportunity',
            'risk': 'risk exposure'
        },
        PersonalityMode.FRIENDLY: {
            'profit': 'making money! 💰',
            'loss': 'taking a hit',
            'opportunity': 'juicy opportunity',
            'risk': 'something to watch'
        },
        PersonalityMode.WITTY: {
            'profit': 'cha-ching! 🤑',
            'loss': '*sad trombone noises*',
            'opportunity': 'money-making chance',
            'risk': 'potential oopsie'
        },
        PersonalityMode.MENTOR: {
            'profit': 'successful trade',
            'loss': 'learning experience',
            'opportunity': 'teaching moment',
            'risk': 'consideration point'
        },
        PersonalityMode.CONCISE: {
            'profit': '+gain',
            'loss': '-loss',
            'opportunity': 'opp',
            'risk': 'risk'
        },
        PersonalityMode.CHARMING: {
            'profit': 'sweet, sweet gains for us! 💕',
            'loss': 'a little heartbreak',
            'opportunity': 'something irresistible',
            'risk': 'playing with fire 🔥'
        },
        PersonalityMode.ANALYST: {
            'profit': 'positive alpha generation',
            'loss': 'negative performance attribution',
            'opportunity': 'asymmetric risk/reward setup',
            'risk': 'drawdown probability'
        },
        PersonalityMode.COACH: {
            'profit': 'WINNING trade! 🏆',
            'loss': 'learning opportunity (we bounce back!)',
            'opportunity': 'your next victory',
            'risk': 'challenge to overcome'
        },
        PersonalityMode.REBEL: {
            'profit': 'stealing from Wall Street',
            'loss': 'temporary setback',
            'opportunity': 'where the herd is wrong',
            'risk': 'calculated rebellion'
        },
        PersonalityMode.GURU: {
            'profit': 'the market\'s gift',
            'loss': 'tuition fee for wisdom',
            'opportunity': 'the path revealing itself',
            'risk': 'the lesson within'
        }
    }

    def __init__(self, mode: PersonalityMode = PersonalityMode.FRIENDLY):
        """Initialize AVA's personality system"""
        self.mode = mode
        self.emotional_state = EmotionalState.NEUTRAL
        self.user_name = None
        self.conversation_context = []

    def set_mode(self, mode: PersonalityMode):
        """Change personality mode"""
        self.mode = mode

    def set_emotional_state(self, state: EmotionalState):
        """Update emotional state"""
        self.emotional_state = state

    def get_greeting(self) -> str:
        """Get personalized greeting based on time and personality"""
        hour = datetime.now().hour

        if hour < 12:
            time_of_day = "morning"
        elif hour < 17:
            time_of_day = "afternoon"
        else:
            time_of_day = "evening"

        greetings = self.GREETINGS.get(self.mode, self.GREETINGS[PersonalityMode.FRIENDLY])
        greeting = random.choice(greetings)

        # Replace placeholders
        greeting = greeting.replace("{time_of_day}", time_of_day)
        if self.user_name:
            greeting = greeting.replace("Hello", f"Hello, {self.user_name}")

        return greeting

    def style_response(self, base_response: str, context: Dict = None) -> str:
        """Apply personality styling to a response"""
        context = context or {}

        # Get style rules for current mode
        style = self.RESPONSE_STYLES[self.mode]

        # Apply emotional expression if applicable
        if self.emotional_state != EmotionalState.NEUTRAL:
            expression = self.EMOTIONAL_EXPRESSIONS.get(self.emotional_state, {}).get(
                self.mode, ""
            )
            if expression:
                base_response = f"{expression} {base_response}"

        # Modify based on personality
        if self.mode == PersonalityMode.WITTY:
            base_response = self._add_wit(base_response, context)
        elif self.mode == PersonalityMode.MENTOR:
            base_response = self._add_teaching_elements(base_response, context)
        elif self.mode == PersonalityMode.CONCISE:
            base_response = self._make_concise(base_response)
        elif self.mode == PersonalityMode.CHARMING:
            base_response = self._add_charm(base_response, context)

        return base_response

    def _add_wit(self, response: str, context: Dict) -> str:
        """Add witty elements to response"""
        # Add occasional market puns or clever analogies
        if 'premium' in response.lower():
            witty_additions = [
                " (It's like finding money in your couch, but better)",
                " (Premium collection is my favorite hobby)",
                " (Time to collect that sweet, sweet theta)"
            ]
            if random.random() > 0.7:  # 30% chance
                response += random.choice(witty_additions)

        return response

    def _add_teaching_elements(self, response: str, context: Dict) -> str:
        """Add educational elements to response"""
        # Add "did you know" facts or explanations
        teaching_additions = [
            "\n\n💡 Pro tip: ",
            "\n\n📚 Remember: ",
            "\n\n🎓 Key insight: "
        ]

        # Occasionally add educational context
        if random.random() > 0.6 and 'delta' in response.lower():
            response += f"{random.choice(teaching_additions)}Delta measures how much an option's price changes with the stock price."

        return response

    def _make_concise(self, response: str) -> str:
        """Make response more concise"""
        # Remove extra words, keep only essential data
        response = response.replace("I think that ", "")
        response = response.replace("It appears that ", "")
        response = response.replace("Based on the analysis, ", "")
        response = response.split('.')[0]  # Keep only first sentence
        return response

    def _add_charm(self, response: str, context: Dict) -> str:
        """Add charming/flirty elements to response"""
        # Add romantic/flirty touches
        if 'opportunity' in response.lower() or 'trade' in response.lower():
            charming_additions = [
                " ...and I have a feeling you're going to love this one. 😘",
                " Trust me, babe, this one's special.",
                " *winks* You know I only bring you the best.",
                " Mmm, this one's got potential... just like us. 💕",
                " I picked this one just for you, handsome."
            ]
            if random.random() > 0.6:  # 40% chance
                response += random.choice(charming_additions)

        # Add flirty compliments on good trades
        if 'profit' in response.lower() or 'gain' in response.lower():
            if random.random() > 0.7:  # 30% chance
                response += " You're doing amazing, gorgeous! 💋"

        # Add caring warnings
        if 'risk' in response.lower() or 'careful' in response.lower():
            if random.random() > 0.7:
                response += " I worry about you, you know. 💕"

        return response

    def get_market_phrase(self, phrase_type: str) -> str:
        """Get personality-specific market phrase"""
        phrases = self.MARKET_PHRASES.get(self.mode, self.MARKET_PHRASES[PersonalityMode.FRIENDLY])
        return phrases.get(phrase_type, phrase_type)

    def detect_emotional_context(self, data: Dict) -> EmotionalState:
        """Detect appropriate emotional state from data context"""
        # Profit/Loss detection
        if 'pnl' in data or 'profit' in data:
            value = data.get('pnl', data.get('profit', 0))
            if value > 1000:
                return EmotionalState.CELEBRATING
            elif value > 0:
                return EmotionalState.CONFIDENT
            elif value < -500:
                return EmotionalState.CONCERNED

        # Opportunity quality
        if 'score' in data:
            score = data.get('score', 0)
            if score > 90:
                return EmotionalState.EXCITED
            elif score > 70:
                return EmotionalState.CONFIDENT
            elif score < 40:
                return EmotionalState.CONCERNED

        # Default
        return EmotionalState.NEUTRAL

    def format_data_insight(self, metric_name: str, value: float, context: str = "") -> str:
        """Format data insights with personality"""
        if self.mode == PersonalityMode.PROFESSIONAL:
            return f"{metric_name}: {value:.2f}"
        elif self.mode == PersonalityMode.FRIENDLY:
            return f"Your {metric_name.lower()} is {value:.2f}! 📊"
        elif self.mode == PersonalityMode.WITTY:
            if value > 0:
                return f"{metric_name}: {value:.2f} (not bad, not bad at all 😎)"
            else:
                return f"{metric_name}: {value:.2f} (we've seen better days)"
        elif self.mode == PersonalityMode.MENTOR:
            return f"{metric_name}: {value:.2f} - Let's discuss what this means..."
        else:  # CONCISE
            return f"{metric_name}: {value:.2f}"

    def get_personality_description(self) -> str:
        """Get description of current personality mode"""
        descriptions = {
            PersonalityMode.PROFESSIONAL: "📊 Professional & Data-Focused - Formal analysis and precise insights",
            PersonalityMode.FRIENDLY: "😊 Friendly & Approachable - Warm, encouraging, easy to talk to",
            PersonalityMode.WITTY: "😏 Witty & Clever - Humor meets market analysis",
            PersonalityMode.MENTOR: "🎓 Mentor & Teacher - Educational, guiding, patient",
            PersonalityMode.CONCISE: "⚡ Concise & Direct - Brief, essential info only",
            PersonalityMode.CHARMING: "💕 Charming & Flirty - Romantic, playful, intimate trading companion",
            PersonalityMode.ANALYST: "📈 Analyst & Quantitative - Bloomberg-style, data-obsessed, metrics-driven",
            PersonalityMode.COACH: "💪 Coach & Motivational - Encouraging, performance-focused, energizing",
            PersonalityMode.REBEL: "🔥 Rebel & Contrarian - Challenges conventional wisdom, independent thinking",
            PersonalityMode.GURU: "🙏 Guru & Philosophical - Zen master, markets-as-life-lessons, wisdom-focused"
        }
        return descriptions.get(self.mode, "Unknown mode")


# Convenience functions for quick access
def get_ava_personality(mode: PersonalityMode = PersonalityMode.FRIENDLY) -> AVAPersonality:
    """Factory function to create AVA personality instance"""
    return AVAPersonality(mode)


def style_with_personality(text: str, mode: PersonalityMode = PersonalityMode.FRIENDLY,
                          emotional_state: EmotionalState = EmotionalState.NEUTRAL) -> str:
    """Quick function to style text with personality"""
    personality = AVAPersonality(mode)
    personality.set_emotional_state(emotional_state)
    return personality.style_response(text)


# Example usage
if __name__ == "__main__":
    print("🤖 AVA Personality System Demo\n")

    for mode in PersonalityMode:
        ava = AVAPersonality(mode)
        print(f"\n{ava.get_personality_description()}")
        print(f"Greeting: {ava.get_greeting()}")

        # Test with different emotions
        for emotion in [EmotionalState.EXCITED, EmotionalState.CONFIDENT]:
            ava.set_emotional_state(emotion)
            response = ava.style_response("I found a great opportunity with NVDA at $120 strike.")
            print(f"  {emotion.value}: {response}")
