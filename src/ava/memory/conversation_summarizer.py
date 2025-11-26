"""
Conversation Summarizer
=======================

Intelligent conversation summarization for context compression and cost optimization.

Features:
- Automatic conversation summarization
- Vector embedding generation for semantic search
- Entity and topic extraction
- Sentiment analysis
- Token compression (up to 90% reduction)

Author: Magnus Trading Platform
Created: 2025-11-23
"""

import re
import openai
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from loguru import logger

from src.services.llm_service import get_llm_service
from src.ava.memory.memory_manager import ConversationSummary


class ConversationSummarizer:
    """
    Intelligent conversation summarizer with vector embeddings

    Compresses long conversations into concise summaries while preserving
    key information for context and semantic search.
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize Conversation Summarizer

        Args:
            openai_api_key: OpenAI API key for embeddings (optional)
        """
        self.llm_service = get_llm_service()
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
        logger.info("Conversation Summarizer initialized")

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Rough estimate: 1 token ≈ 4 characters or 0.75 words
        return int(len(text.split()) * 1.3)

    def extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from text (tickers, strategies, etc.)

        Args:
            text: Input text

        Returns:
            List of extracted entities
        """
        entities = set()

        # Extract ticker symbols (2-5 uppercase letters)
        tickers = re.findall(r'\b[A-Z]{2,5}\b', text)
        entities.update(tickers)

        # Extract common trading terms
        trading_terms = [
            'call', 'put', 'spread', 'iron condor', 'butterfly',
            'straddle', 'strangle', 'covered call', 'cash secured put',
            'wheel strategy', 'credit spread', 'debit spread',
            'calendar spread', 'diagonal spread'
        ]

        text_lower = text.lower()
        for term in trading_terms:
            if term in text_lower:
                entities.add(term)

        return list(entities)

    def extract_topics(self, text: str) -> List[str]:
        """
        Extract key topics from text

        Args:
            text: Input text

        Returns:
            List of key topics
        """
        topics = []

        # Define topic keywords
        topic_keywords = {
            'options': ['option', 'call', 'put', 'strike', 'expiration'],
            'strategy': ['strategy', 'plan', 'approach', 'method'],
            'analysis': ['analysis', 'analyze', 'examine', 'evaluate'],
            'portfolio': ['portfolio', 'holdings', 'positions'],
            'risk': ['risk', 'hedge', 'protect', 'exposure'],
            'earnings': ['earnings', 'report', 'results'],
            'market': ['market', 'trend', 'sentiment'],
            'technical': ['technical', 'chart', 'indicator', 'pattern'],
            'fundamental': ['fundamental', 'valuation', 'pe ratio']
        }

        text_lower = text.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)

        return topics

    def analyze_sentiment(self, text: str) -> str:
        """
        Analyze sentiment of conversation

        Args:
            text: Input text

        Returns:
            Sentiment: 'positive', 'negative', 'neutral', or 'mixed'
        """
        text_lower = text.lower()

        # Positive indicators
        positive_words = [
            'good', 'great', 'excellent', 'bullish', 'profit', 'gain',
            'success', 'opportunity', 'positive', 'confident'
        ]
        positive_count = sum(1 for word in positive_words if word in text_lower)

        # Negative indicators
        negative_words = [
            'bad', 'poor', 'loss', 'bearish', 'risk', 'concern',
            'problem', 'negative', 'worried', 'decline'
        ]
        negative_count = sum(1 for word in negative_words if word in text_lower)

        # Determine sentiment
        if positive_count > negative_count * 1.5:
            return 'positive'
        elif negative_count > positive_count * 1.5:
            return 'negative'
        elif positive_count > 0 and negative_count > 0:
            return 'mixed'
        else:
            return 'neutral'

    def summarize_conversation(
        self,
        messages: List[Dict[str, str]],
        max_summary_tokens: int = 200,
        include_entities: bool = True
    ) -> Tuple[str, List[str], List[str]]:
        """
        Summarize a conversation

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_summary_tokens: Maximum tokens for summary
            include_entities: Whether to extract entities

        Returns:
            Tuple of (summary, topics, entities)
        """
        # Combine conversation into single text
        conversation_text = "\n".join([
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in messages
        ])

        # Create summarization prompt
        prompt = f"""Summarize the following trading/investment conversation in {max_summary_tokens} tokens or less.
Focus on:
- Key questions asked
- Main topics discussed
- Important decisions or recommendations
- Specific tickers or strategies mentioned

Conversation:
{conversation_text[:4000]}  # Limit input length

Summary:"""

        try:
            # Generate summary using local LLM (Ollama) for cost efficiency
            result = self.llm_service.generate(
                prompt,
                provider="ollama",  # Use local model
                max_tokens=max_summary_tokens,
                temperature=0.3  # Low temperature for factual summary
            )

            summary = result['text'].strip()

            # Extract topics and entities
            topics = self.extract_topics(conversation_text)
            entities = self.extract_entities(conversation_text) if include_entities else []

            logger.info(f"Summarized conversation: {len(conversation_text)} chars → {len(summary)} chars")

            return summary, topics, entities

        except Exception as e:
            logger.error(f"Failed to summarize conversation: {e}")
            # Fallback: simple truncation
            truncated = conversation_text[:max_summary_tokens * 4]
            return truncated, [], []

    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate vector embedding for text using OpenAI

        Args:
            text: Input text

        Returns:
            1536-dimensional embedding vector or None if failed
        """
        if not self.openai_api_key:
            logger.warning("No OpenAI API key provided, skipping embedding generation")
            return None

        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text[:8000]  # Limit input length
            )

            embedding = response['data'][0]['embedding']
            return np.array(embedding, dtype=np.float32)

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    def create_conversation_summary(
        self,
        messages: List[Dict[str, str]],
        conversation_start: datetime,
        conversation_end: datetime,
        generate_embedding: bool = True
    ) -> Tuple[ConversationSummary, Optional[np.ndarray]]:
        """
        Create a complete conversation summary with embedding

        Args:
            messages: List of message dictionaries
            conversation_start: Conversation start time
            conversation_end: Conversation end time
            generate_embedding: Whether to generate vector embedding

        Returns:
            Tuple of (ConversationSummary, embedding vector)
        """
        # Combine messages
        conversation_text = "\n".join([
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in messages
        ])

        # Estimate tokens
        original_tokens = self.estimate_tokens(conversation_text)

        # Generate summary
        summary_text, topics, entities = self.summarize_conversation(
            messages,
            max_summary_tokens=200
        )

        summary_tokens = self.estimate_tokens(summary_text)

        # Analyze sentiment
        sentiment = self.analyze_sentiment(conversation_text)

        # Generate embedding
        embedding = None
        if generate_embedding and self.openai_api_key:
            embedding = self.generate_embedding(summary_text)

        # Create summary object
        summary = ConversationSummary(
            summary=summary_text,
            key_topics=topics,
            entities_mentioned=entities,
            message_count=len(messages),
            conversation_start=conversation_start,
            conversation_end=conversation_end,
            original_tokens=original_tokens,
            summary_tokens=summary_tokens,
            sentiment=sentiment,
            model_used="llama3.1:8b"  # Using local Ollama model
        )

        logger.info(f"Created summary: {original_tokens} → {summary_tokens} tokens "
                   f"({summary.summary_tokens/summary.original_tokens*100:.1f}% of original)")

        return summary, embedding

    def batch_summarize_conversations(
        self,
        conversation_groups: List[List[Dict[str, str]]],
        generate_embeddings: bool = True
    ) -> List[Tuple[ConversationSummary, Optional[np.ndarray]]]:
        """
        Batch process multiple conversations

        Args:
            conversation_groups: List of conversation message lists
            generate_embeddings: Whether to generate embeddings

        Returns:
            List of (ConversationSummary, embedding) tuples
        """
        results = []

        for i, messages in enumerate(conversation_groups):
            try:
                # Use current time as start/end for now
                # In practice, these would come from message timestamps
                now = datetime.now()

                summary, embedding = self.create_conversation_summary(
                    messages,
                    conversation_start=now,
                    conversation_end=now,
                    generate_embedding=generate_embeddings
                )

                results.append((summary, embedding))

                logger.info(f"Processed conversation {i+1}/{len(conversation_groups)}")

            except Exception as e:
                logger.error(f"Failed to process conversation {i+1}: {e}")
                continue

        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def summarize_recent_messages(
    messages: List[Dict[str, str]],
    max_messages: int = 10
) -> str:
    """
    Quick summary of recent messages

    Args:
        messages: List of message dicts
        max_messages: Maximum messages to include

    Returns:
        Summary string
    """
    summarizer = ConversationSummarizer()

    # Take last N messages
    recent = messages[-max_messages:]

    summary, _, _ = summarizer.summarize_conversation(recent, max_summary_tokens=100)

    return summary
