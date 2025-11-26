# AVA Quick-Start Integration Guide

**Ready-to-use code examples for integrating enhanced AVA features**

---

## Setup (One-Time)

### 1. Run Database Migration
```bash
# Make sure PostgreSQL is running
psql -U your_user -d your_database -f src/ava/migrations/002_memory_system.sql
```

### 2. Set Environment Variables
```bash
# Add to .env
DATABASE_URL=postgresql://user:password@localhost:5432/magnus
OPENAI_API_KEY=sk-...  # Optional, for embeddings
```

### 3. Verify Ollama is Running
```bash
ollama list
# Should show: llama3.1:8b, mistral:7b, etc.
```

---

## Usage Examples

### Example 1: Basic LLM Query (Uses Local Ollama)
```python
from src.services.llm_service import get_llm_service

# Get LLM service (automatically uses Ollama for simple queries)
llm = get_llm_service()

# Generate response
result = llm.generate(
    "What is a covered call strategy?",
    max_tokens=200,
    temperature=0.7
)

print(f"Provider: {result['provider']}")  # ollama
print(f"Model: {result['model']}")        # llama3.1:8b
print(f"Cost: ${result['cost']:.6f}")     # $0.000000
print(f"Response: {result['text']}")
```

### Example 2: Store User Preference
```python
from src.ava.memory import get_memory_manager
import os

# Initialize memory manager
memory = get_memory_manager(os.getenv("DATABASE_URL"))

# Store user preference
memory.update_user_preference(
    user_id="telegram:123456",
    platform="telegram",
    key="risk_tolerance",
    value="moderate",
    importance=8
)

# Later, retrieve it
risk_tolerance = memory.get_user_preference(
    user_id="telegram:123456",
    platform="telegram",
    key="risk_tolerance",
    default="conservative"
)

print(f"User risk tolerance: {risk_tolerance}")  # moderate
```

### Example 3: Track Ticker Mentions
```python
from src.ava.memory import get_memory_manager, EntityMemory
import os

memory = get_memory_manager(os.getenv("DATABASE_URL"))

# User mentions AAPL
entity = EntityMemory(
    entity_type="ticker",
    entity_id="AAPL",
    entity_name="Apple Inc.",
    sentiment="positive",
    interest_score=8,
    tags=["tech", "options"]
)

memory.track_entity(
    user_id="telegram:123456",
    platform="telegram",
    entity=entity,
    context="User asked about AAPL covered calls"
)

# Get user's top tickers
top_tickers = memory.get_user_entities(
    user_id="telegram:123456",
    platform="telegram",
    entity_type="ticker",
    min_mentions=2
)

print(f"User's favorite tickers: {[e['entity_id'] for e in top_tickers]}")
```

### Example 4: Summarize Conversation
```python
from src.ava.memory.conversation_summarizer import ConversationSummarizer
from src.ava.memory import get_memory_manager
from datetime import datetime
import os

# Initialize
summarizer = ConversationSummarizer(openai_api_key=os.getenv("OPENAI_API_KEY"))
memory = get_memory_manager(os.getenv("DATABASE_URL"))

# Conversation messages
messages = [
    {"role": "user", "content": "What's a good wheel strategy for AAPL?"},
    {"role": "assistant", "content": "The wheel strategy for AAPL involves..."},
    {"role": "user", "content": "What strike price should I use?"},
    {"role": "assistant", "content": "For AAPL, consider..."},
    # ... more messages
]

# Create summary with embedding
summary, embedding = summarizer.create_conversation_summary(
    messages=messages,
    conversation_start=datetime.now(),
    conversation_end=datetime.now(),
    generate_embedding=True
)

# Store in memory
memory.store_conversation_summary(
    user_id="telegram:123456",
    platform="telegram",
    summary=summary,
    embedding=embedding
)

print(f"Summary: {summary.summary}")
print(f"Topics: {summary.key_topics}")
print(f"Entities: {summary.entities_mentioned}")
print(f"Token savings: {summary.original_tokens - summary.summary_tokens}")
```

### Example 5: Search Similar Past Conversations
```python
from src.ava.memory.conversation_summarizer import ConversationSummarizer
from src.ava.memory import get_memory_manager
import os

summarizer = ConversationSummarizer(openai_api_key=os.getenv("OPENAI_API_KEY"))
memory = get_memory_manager(os.getenv("DATABASE_URL"))

# User asks new question
query = "Tell me about calendar spreads"

# Generate embedding for query
query_embedding = summarizer.generate_embedding(query)

# Search similar past conversations
similar = memory.search_similar_conversations(
    user_id="telegram:123456",
    platform="telegram",
    query_embedding=query_embedding,
    limit=3,
    similarity_threshold=0.7
)

print("Similar past conversations:")
for conv in similar:
    print(f"- {conv['summary']} (similarity: {conv['similarity']:.2f})")
    print(f"  Topics: {conv['key_topics']}")
    print()
```

### Example 6: Complete AVA Integration
```python
import os
from datetime import datetime
from src.services.llm_service import get_llm_service
from src.ava.memory import get_memory_manager, EntityMemory
from src.ava.memory.conversation_summarizer import ConversationSummarizer

class EnhancedAVA:
    """AVA with memory and local LLM"""

    def __init__(self):
        self.llm = get_llm_service()
        self.memory = get_memory_manager(os.getenv("DATABASE_URL"))
        self.summarizer = ConversationSummarizer(os.getenv("OPENAI_API_KEY"))
        self.conversation_messages = []

    def process_message(self, user_id: str, platform: str, message: str) -> str:
        """Process user message with memory context"""

        # 1. Get user context from memory
        preferences = self.memory.get_user_memory(
            user_id=user_id,
            platform=platform,
            memory_type="preference"
        )

        # 2. Get recent conversation summaries for context
        recent_summaries = self.memory.get_recent_summaries(
            user_id=user_id,
            platform=platform,
            limit=3
        )

        # 3. Build context-aware prompt
        context = f"User preferences: {preferences}\n"
        context += f"Recent discussions: {[s['summary'] for s in recent_summaries]}\n"
        full_prompt = f"{context}\n\nUser: {message}\nAssistant:"

        # 4. Generate response (automatically uses local LLM for simple queries)
        result = self.llm.generate(
            full_prompt,
            max_tokens=500,
            temperature=0.7
        )

        response = result['text']

        # 5. Track conversation
        self.conversation_messages.append({
            "role": "user",
            "content": message
        })
        self.conversation_messages.append({
            "role": "assistant",
            "content": response
        })

        # 6. Extract and track entities
        entities = self.summarizer.extract_entities(message)
        for entity_id in entities:
            if entity_id.isupper() and len(entity_id) <= 5:  # Likely a ticker
                entity = EntityMemory(
                    entity_type="ticker",
                    entity_id=entity_id,
                    sentiment=self.summarizer.analyze_sentiment(message)
                )
                self.memory.track_entity(
                    user_id=user_id,
                    platform=platform,
                    entity=entity,
                    context=message[:200]
                )

        # 7. Summarize if conversation is getting long
        if len(self.conversation_messages) >= 10:
            summary, embedding = self.summarizer.create_conversation_summary(
                messages=self.conversation_messages,
                conversation_start=datetime.now(),
                conversation_end=datetime.now(),
                generate_embedding=True
            )

            self.memory.store_conversation_summary(
                user_id=user_id,
                platform=platform,
                summary=summary,
                embedding=embedding
            )

            # Reset conversation
            self.conversation_messages = []

        return response


# Usage
ava = EnhancedAVA()
response = ava.process_message(
    user_id="telegram:123456",
    platform="telegram",
    message="What's a good strategy for TSLA this week?"
)
print(response)
```

### Example 7: Memory Statistics
```python
from src.ava.memory import get_memory_manager
import os

memory = get_memory_manager(os.getenv("DATABASE_URL"))

# Get comprehensive memory stats
stats = memory.get_memory_stats(
    user_id="telegram:123456",
    platform="telegram"
)

print("Memory Statistics:")
print(f"User memories: {stats['memory_summary']}")
print(f"Tracked entities: {stats['entity_stats']}")
print(f"Conversations: {stats['conversation_stats']}")
```

### Example 8: Intelligent Routing Test
```python
from src.services.llm_service import get_llm_service

llm = get_llm_service()

# Test different query complexities
queries = [
    ("Hi, how are you?", "TRIVIAL"),
    ("What is the current price of AAPL?", "SIMPLE"),
    ("Explain the differences between credit and debit spreads", "MODERATE"),
    ("Design an optimal delta-neutral options strategy for high IV", "COMPLEX"),
    ("Write a Python function to calculate Black-Scholes pricing", "ADVANCED"),
]

for query, expected_complexity in queries:
    result = llm.generate(query, max_tokens=100)

    print(f"\nQuery: {query}")
    print(f"Expected: {expected_complexity}")
    print(f"Provider: {result['provider']}")
    print(f"Model: {result['model']}")
    print(f"Cost: ${result['cost']:.6f}")

# Get routing statistics
routing_stats = llm.get_routing_stats()
print(f"\n\nRouting Statistics:")
print(f"Free tier percentage: {routing_stats['free_tier_percentage']}%")
print(f"Total cost: ${routing_stats['actual_cost']:.4f}")
print(f"Savings vs premium: ${routing_stats['savings']:.4f}")
```

---

## Testing Checklist

- [ ] Run database migration successfully
- [ ] Verify Ollama is running and models are available
- [ ] Test basic LLM query (should use Ollama, cost $0)
- [ ] Store and retrieve user preference
- [ ] Track entity mention
- [ ] Generate conversation summary
- [ ] Search similar conversations (requires OpenAI API key)
- [ ] Check memory statistics
- [ ] Verify intelligent routing (70-80% FREE tier)

---

## Common Patterns

### Pattern 1: Context-Aware Responses
```python
# Always load user context before generating responses
preferences = memory.get_user_memory(user_id, platform, "preference")
recent_summaries = memory.get_recent_summaries(user_id, platform, limit=3)
# Include in prompt
```

### Pattern 2: Progressive Memory Building
```python
# Learn from every interaction
1. Extract entities → track_entity()
2. Infer preferences → update_user_preference()
3. Summarize periodically → store_conversation_summary()
```

### Pattern 3: Cost Optimization
```python
# Let intelligent routing decide provider
result = llm.generate(prompt)  # Automatic routing

# Force free tier for non-critical
result = llm.generate(prompt, provider="ollama")

# Check routing effectiveness
stats = llm.get_routing_stats()
assert stats['free_tier_percentage'] >= 70  # Should be 70-80%
```

### Pattern 4: Error Handling
```python
try:
    result = llm.generate_with_fallback(prompt)
    # Automatically tries: ollama → groq → deepseek → ...
except Exception as e:
    logger.error(f"All providers failed: {e}")
    # Fallback to cached response or error message
```

---

## Performance Tips

1. **Use Ollama for simple queries** - 10-30x faster than cloud
2. **Summarize long conversations** - Save up to 90% tokens
3. **Cache user preferences** - Avoid repeated database queries
4. **Batch entity tracking** - Update multiple entities at once
5. **Generate embeddings async** - Don't block on OpenAI API

---

## Troubleshooting

### Ollama not working?
```bash
ollama serve  # Start Ollama server
ollama list   # Verify models
```

### Memory queries failing?
```bash
# Check database connection
psql -U user -d magnus -c "SELECT COUNT(*) FROM ava_user_memory;"
```

### High costs?
```python
# Check routing stats
llm.get_routing_stats()
# Should show 70-80% FREE tier usage
```

---

## Next Steps

1. ✅ Integrate examples into your AVA core
2. ✅ Run tests to verify everything works
3. 🔄 Deploy to production
4. 📊 Monitor cost savings and performance
5. 🚀 Add Discord, webhooks, advanced RAG

**You're ready to go! 🎉**
