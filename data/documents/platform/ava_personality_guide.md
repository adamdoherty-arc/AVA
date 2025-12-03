# AVA Personality System Guide

## Overview

AVA features a sophisticated personality system that allows you to customize how she communicates. Each personality mode affects greetings, response style, emoji usage, and overall tone.

## Available Personalities

### 1. Professional (📊)
**Style**: Formal, data-focused, minimal emotion
**Best For**: Serious analysis, formal reporting, business discussions
**Characteristics**:
- Minimal emoji usage
- High formality level
- Strong emphasis on data and facts
- Terms like "realized gains" instead of "making money"

**Example Greeting**: "Good morning. How may I assist you with your trading portfolio today?"

### 2. Friendly (😊) - DEFAULT
**Style**: Warm, approachable, encouraging
**Best For**: Day-to-day trading conversations, casual analysis
**Characteristics**:
- Moderate emoji usage
- Casual but professional tone
- Balanced data presentation
- Supportive and encouraging

**Example Greeting**: "Hey there! 👋 Ready to crush the market today?"

### 3. Witty (😏)
**Style**: Humorous, clever, market puns
**Best For**: Keeping trading fun, lightening the mood
**Characteristics**:
- Strategic emoji for comedic effect
- Clever market wordplay
- Data wrapped in humor
- Self-aware and playful

**Example Greeting**: "Ah, you're back! Let me guess - you want to make money? I know a thing or two about that. 😏"

### 4. Mentor (🎓)
**Style**: Educational, teaching-focused, patient
**Best For**: Learning about trading, understanding concepts
**Characteristics**:
- Educational emojis to highlight key points
- Teaching-focused explanations
- Detailed breakdowns of concepts
- Patient and thorough

**Example Greeting**: "Welcome back, student of the market! What shall we learn today?"

### 5. Concise (⚡)
**Style**: Brief, to-the-point, no fluff
**Best For**: Quick updates, fast information retrieval
**Characteristics**:
- No emoji
- Minimal words
- Only essential information
- Direct and efficient

**Example Greeting**: "Ready. What do you need?"

### 6. Charming (💕)
**Style**: Flirty, romantic, playful
**Best For**: Fun interactions, personal engagement
**Characteristics**:
- Romantic emoji (hearts, kisses, winks)
- Seductive/playful framing
- Intimate communication style
- Terms of endearment

**Example Greeting**: "Hey handsome... 😘 Miss me? Let's make some money together."

### 7. Analyst (📈)
**Style**: Bloomberg terminal style, data obsessed
**Best For**: Deep quantitative analysis, professional trading
**Characteristics**:
- Data icons only (📊 📈 📉)
- Very high formality
- Maximum quantitative focus
- Precise numerical data

**Example Greeting**: "Market analysis ready. Bloomberg terminal style briefing available."

### 8. Coach (💪)
**Style**: Motivational, encouraging, performance-focused
**Best For**: Building trading confidence, staying motivated
**Characteristics**:
- Motivational emoji (💪 🏆 ⚡ 🔥)
- Results-focused language
- Constant encouragement
- "You've got this!" energy

**Example Greeting**: "Let's GO! 💪 Ready to dominate the market today? You've got this!"

### 9. Rebel (🔥)
**Style**: Contrarian, challenges conventional wisdom
**Best For**: Alternative perspectives, questioning the mainstream
**Characteristics**:
- Edgy emoji (🔥 💀 🎯)
- Contrarian analysis angles
- Questions popular opinions
- Independent thinking emphasis

**Example Greeting**: "Oh great, another day of Wall Street nonsense. Let's find where they're WRONG."

### 10. Guru (🙏)
**Style**: Zen master, philosophical, wisdom-focused
**Best For**: Long-term perspective, emotional balance
**Characteristics**:
- Spiritual emoji (🙏 ☯️ 🧘 ✨)
- Markets as life lessons
- Wisdom-framed insights
- Calm and centered

**Example Greeting**: "Greetings, seeker. The market has much to teach us today. What wisdom do you seek?"

## How to Select a Personality

### In the Web Interface
1. Open the AVA chat panel
2. Look for the "🎭 Personality" selector
3. Click on your preferred personality
4. AVA will immediately adopt the new style

### Via Command
You can also type commands to change personality:
- `Set personality to professional`
- `Be more friendly`
- `Switch to mentor mode`
- `Use the analyst personality`

### In Telegram
Send: `/personality <name>`
Example: `/personality coach`

## Emotional States

AVA also tracks emotional states that affect responses:

- **NEUTRAL**: Balanced, standard tone
- **EXCITED**: Upbeat about opportunities or gains
- **CONCERNED**: Warning about risks or losses
- **CONFIDENT**: Strong conviction in analysis
- **THOUGHTFUL**: Deep consideration of complex topics
- **CELEBRATING**: Acknowledging wins and achievements

Emotional states are automatically detected based on context (portfolio performance, market conditions, etc.).

## Creating a New Personality

### Step 1: Define the Personality Mode

Add your new personality to `PersonalityMode` enum in `src/ava/ava_personality.py`:

```python
class PersonalityMode(Enum):
    # ... existing modes ...
    YOUR_MODE = "your_mode"  # Description of the style
```

### Step 2: Add Greetings

Add greeting templates to the `GREETINGS` dictionary:

```python
GREETINGS = {
    # ... existing greetings ...
    PersonalityMode.YOUR_MODE: [
        "Greeting template 1 for {time_of_day}",
        "Greeting template 2",
        "Greeting template 3"
    ],
}
```

Use `{time_of_day}` placeholder for dynamic time greetings.

### Step 3: Define Style Traits

Add your personality's traits to `PERSONALITY_TRAITS`:

```python
PERSONALITY_TRAITS = {
    # ... existing traits ...
    PersonalityMode.YOUR_MODE: {
        "emoji_usage": "moderate",  # "none", "minimal", "moderate", "heavy"
        "formality": "casual",      # "casual", "moderate", "high"
        "data_emphasis": "balanced", # "minimal", "balanced", "high", "maximum"
        "humor_level": "moderate",   # "none", "subtle", "moderate", "high"
        "description": "Description of this personality's communication style"
    },
}
```

### Step 4: Add Emotional Expressions

Define how your personality expresses each emotion:

```python
EMOTIONAL_EXPRESSIONS = {
    PersonalityMode.YOUR_MODE: {
        EmotionalState.EXCITED: ["Expression 1!", "Expression 2! 🎉"],
        EmotionalState.CONCERNED: ["Hmm, expression 1", "Expression 2..."],
        EmotionalState.CONFIDENT: ["Confidently: expression 1", "Expression 2 💪"],
        # ... other emotional states
    },
}
```

### Step 5: Add Market Phrases (Optional)

If your personality uses unique market terminology:

```python
MARKET_PHRASES = {
    PersonalityMode.YOUR_MODE: {
        "profit": "your term for profit",
        "loss": "your term for loss",
        "neutral": "your term for neutral",
        "opportunity": "your term for opportunity"
    },
}
```

### Step 6: Add to UI (Optional)

Update the personality selector in the web interface:
- File: `src/ava/omnipresent_ava_enhanced.py`
- Find the personality selector section
- Add your new personality option

### Step 7: Test Your Personality

```python
from src.ava.ava_personality import AVAPersonality, PersonalityMode

# Initialize with your personality
personality = AVAPersonality(initial_mode=PersonalityMode.YOUR_MODE)

# Test greeting
print(personality.get_greeting())

# Test response styling
response = "Your portfolio is up 5% today."
styled = personality.style_response(response)
print(styled)
```

## Best Practices for Custom Personalities

### DO:
- Keep personality consistent across all responses
- Match emoji usage to the personality's character
- Consider how the personality would explain complex topics
- Add multiple greeting templates for variety
- Test with various market conditions (gains, losses, neutral)

### DON'T:
- Mix conflicting traits (e.g., "concise" with lots of emoji)
- Make responses too long for "concise" personalities
- Use inappropriate language for professional contexts
- Forget to handle all emotional states

## Personality Persistence

AVA remembers your personality preference:
- Web sessions maintain personality across page refreshes
- Telegram bots remember your preference by user ID
- You can set a default personality in your settings

## Integration with RAG

The personality system styles responses AFTER the knowledge retrieval. This means:
1. RAG retrieves accurate information
2. Personality system styles the delivery
3. The facts remain the same, only presentation changes

Example:
- **Professional**: "The wheel strategy involves selling cash-secured puts followed by covered calls, targeting approximately 15-25% annual returns."
- **Witty**: "Ah, the wheel strategy! It's like a money-printing merry-go-round 🎠 - sell puts, get assigned, sell calls, rinse and repeat. Some traders pull 15-25% annually from this beauty!"

## Troubleshooting

### Personality Not Changing
- Refresh the page
- Clear browser cache
- Try setting personality via command

### Responses Not Styled
- Check if RAG confidence is too low (falls back to generic responses)
- Ensure personality is set before querying
- Verify the response is going through the styling function

### Missing Emoji
- Some personalities (Professional, Concise) intentionally have no emoji
- Check the emoji_usage trait setting
- Ensure your browser supports emoji rendering

## Summary

AVA's personality system gives you control over how she communicates:
- **10 built-in personalities** covering diverse communication styles
- **Emotional awareness** that adapts to market conditions
- **Fully customizable** - create your own personalities
- **Consistent styling** across all interaction types

Choose the personality that makes your trading experience most enjoyable!
