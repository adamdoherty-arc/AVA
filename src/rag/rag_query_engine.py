"""
RAG Query Engine - Retrieve similar trades and generate recommendations

This module handles:
1. Similarity search in Qdrant
2. Re-ranking retrieved trades
3. Context assembly for LLM
4. Generating recommendations (FREE via Groq/Ollama)

Updated: 2025-11-29 - Now uses FREE LLM providers by default

Author: Magnus Wheel Strategy Dashboard
Created: 2025-11-06
"""

import os
import sys
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from dotenv import load_dotenv
import logging

from src.ava.core.llm_engine import LLMClient, LLMProvider
from src.ava.core.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class RAGQueryEngine:
    """
    RAG-powered query engine for options trading recommendations
    Now uses FREE LLM providers (Groq/Ollama) by default!
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None
    ):
        """
        Initialize RAG query engine

        Args:
            embedding_model: Hugging Face model name
            qdrant_url: Qdrant cluster URL
            qdrant_api_key: Qdrant API key
        """
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize Qdrant client
        self.qdrant_url = qdrant_url or os.getenv('QDRANT_URL')
        self.qdrant_api_key = qdrant_api_key or os.getenv('QDRANT_API_KEY')

        if not self.qdrant_api_key:
            raise ValueError("QDRANT_API_KEY must be set")

        self.qdrant = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )

        self.collection_name = "options_trades"

        # Initialize FREE LLM client using centralized config
        config = get_config()
        provider_map = {
            "ollama": LLMProvider.OLLAMA,
            "groq": LLMProvider.GROQ,
            "huggingface": LLMProvider.HUGGINGFACE,
            "anthropic": LLMProvider.ANTHROPIC,
            "openai": LLMProvider.OPENAI,
        }
        provider = provider_map.get(config.ai.provider.lower(), LLMProvider.GROQ)

        self.llm = LLMClient(
            provider=provider,
            model=config.ai.default_model,
            cache_enabled=True,
            cache_ttl=300
        )
        logger.info(f"LLM initialized with FREE provider: {provider.value}")

        # Re-ranking weights
        self.rerank_weights = {
            'similarity': 0.5,
            'recency': 0.25,
            'outcome': 0.25
        }

    def format_alert_for_embedding(self, alert: Dict[str, Any]) -> str:
        """
        Format new alert for embedding

        Args:
            alert: New alert dictionary

        Returns:
            Formatted text string
        """
        text_parts = []

        text_parts.append(f"Strategy: {alert.get('strategy', 'Unknown')}")
        text_parts.append(f"Ticker: {alert.get('ticker', 'UNKNOWN')}")

        if alert.get('action'):
            text_parts.append(f"Action: {alert['action']}")
        if alert.get('strike_price'):
            text_parts.append(f"Strike: ${alert['strike_price']:.2f}")
        if alert.get('dte'):
            text_parts.append(f"Days to Expiration: {alert['dte']}")
        if alert.get('premium'):
            text_parts.append(f"Premium: ${alert['premium']:.2f}")

        # Current market conditions
        if alert.get('current_vix'):
            text_parts.append(f"VIX: {alert['current_vix']:.1f}")
        if alert.get('iv_rank'):
            text_parts.append(f"IV Rank: {alert['iv_rank']}")

        if alert.get('alert_text'):
            text_parts.append(f"Alert: {alert['alert_text']}")

        return "\n".join(text_parts)

    def build_search_filters(self, alert: Dict[str, Any]) -> Filter:
        """
        Build Qdrant filters based on alert parameters

        Args:
            alert: New alert dictionary

        Returns:
            Qdrant Filter object
        """
        must_conditions = []

        # Always filter for closed trades only
        must_conditions.append(
            FieldCondition(
                key="status",
                match=MatchValue(value="closed")
            )
        )

        # Same ticker (required)
        if alert.get('ticker'):
            must_conditions.append(
                FieldCondition(
                    key="ticker",
                    match=MatchValue(value=alert['ticker'].upper())
                )
            )

        # Same strategy (required)
        if alert.get('strategy'):
            must_conditions.append(
                FieldCondition(
                    key="strategy",
                    match=MatchValue(value=alert['strategy'])
                )
            )

        # Similar DTE (+/- 7 days)
        if alert.get('dte'):
            must_conditions.append(
                FieldCondition(
                    key="dte",
                    range=Range(
                        gte=max(0, alert['dte'] - 7),
                        lte=alert['dte'] + 7
                    )
                )
            )

        # Similar VIX regime (+/- 5 points)
        if alert.get('current_vix'):
            must_conditions.append(
                FieldCondition(
                    key="vix_at_entry",
                    range=Range(
                        gte=max(0, alert['current_vix'] - 5),
                        lte=alert['current_vix'] + 5
                    )
                )
            )

        return Filter(must=must_conditions) if must_conditions else None

    def search_similar_trades(
        self,
        alert: Dict[str, Any],
        limit: int = 10,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar historical trades

        Args:
            alert: New alert dictionary
            limit: Maximum number of results
            score_threshold: Minimum similarity score

        Returns:
            List of similar trades with scores
        """
        try:
            # Format alert for embedding
            alert_text = self.format_alert_for_embedding(alert)

            # Generate embedding
            logger.info(f"Generating embedding for: {alert.get('ticker')} {alert.get('strategy')}")
            embedding = self.embedding_model.encode(alert_text, convert_to_numpy=True)

            # Build filters
            filters = self.build_search_filters(alert)

            # Search Qdrant
            logger.info(f"Searching for similar trades (limit={limit}, threshold={score_threshold})")
            search_results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=embedding.tolist(),
                query_filter=filters,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False
            )

            # Convert to list of dicts
            similar_trades = []
            for result in search_results:
                trade = result.payload
                trade['similarity_score'] = result.score
                similar_trades.append(trade)

            logger.info(f"Found {len(similar_trades)} similar trades")
            return similar_trades

        except Exception as e:
            logger.error(f"Error searching similar trades: {e}")
            return []

    def rerank_trades(
        self,
        trades: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Re-rank trades by multiple factors

        Args:
            trades: List of similar trades
            top_k: Number of top trades to return

        Returns:
            Top-k re-ranked trades
        """
        if not trades:
            return []

        for trade in trades:
            # Recency score (decay over 1 year)
            entry_date = datetime.fromisoformat(trade['entry_date'])
            days_old = (datetime.now() - entry_date).days
            recency_score = max(0, 1 - (days_old / 365))

            # Outcome quality score (normalize P&L percent to 0-1)
            pnl_percent = trade.get('pnl_percent', 0)
            outcome_score = min(1.0, max(0, pnl_percent / 100))

            # Combined score
            combined_score = (
                trade['similarity_score'] * self.rerank_weights['similarity'] +
                recency_score * self.rerank_weights['recency'] +
                outcome_score * self.rerank_weights['outcome']
            )

            trade['combined_score'] = combined_score
            trade['recency_score'] = recency_score
            trade['outcome_score'] = outcome_score

        # Sort by combined score
        ranked_trades = sorted(trades, key=lambda x: x['combined_score'], reverse=True)

        return ranked_trades[:top_k]

    def calculate_statistics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate aggregate statistics from similar trades

        Args:
            trades: List of similar trades

        Returns:
            Statistics dictionary
        """
        if not trades:
            return {
                'count': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'avg_pnl_percent': 0.0,
                'best_trade': None,
                'worst_trade': None,
                'avg_hold_days': 0
            }

        wins = [t for t in trades if t.get('win', False)]
        win_rate = (len(wins) / len(trades)) * 100 if trades else 0

        pnls = [t.get('pnl', 0) for t in trades if t.get('pnl') is not None]
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0

        pnl_percents = [t.get('pnl_percent', 0) for t in trades if t.get('pnl_percent') is not None]
        avg_pnl_percent = sum(pnl_percents) / len(pnl_percents) if pnl_percents else 0

        best_trade = max(trades, key=lambda x: x.get('pnl', 0)) if trades else None
        worst_trade = min(trades, key=lambda x: x.get('pnl', 0)) if trades else None

        hold_days = [t.get('hold_days', 0) for t in trades if t.get('hold_days')]
        avg_hold_days = sum(hold_days) / len(hold_days) if hold_days else 0

        return {
            'count': len(trades),
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'avg_pnl_percent': avg_pnl_percent,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'avg_hold_days': int(avg_hold_days)
        }

    def format_trade_for_context(self, trade: Dict[str, Any], index: int) -> str:
        """
        Format a trade for inclusion in LLM context

        Args:
            trade: Trade dictionary
            index: Trade number

        Returns:
            Formatted trade string
        """
        entry_date = datetime.fromisoformat(trade['entry_date']).strftime('%Y-%m-%d')

        lines = [
            f"### Trade #{index} (Similarity: {trade['similarity_score']:.1%})",
            f"Date: {entry_date}",
            f"Ticker: {trade['ticker']}",
            f"Strategy: {trade['strategy']}",
            f"Strike: ${trade.get('strike_price', 'N/A')} | DTE: {trade.get('dte', 'N/A')}",
            f"Premium: ${trade.get('entry_price', 'N/A')}",
        ]

        # Market conditions
        if trade.get('vix_at_entry'):
            lines.append(f"VIX: {trade['vix_at_entry']:.1f}")
        if trade.get('spy_trend'):
            lines.append(f"Market Trend: {trade['spy_trend']}")

        # Outcome
        outcome = "WIN" if trade.get('win') else "LOSS"
        pnl = trade.get('pnl', 0)
        pnl_pct = trade.get('pnl_percent', 0)
        hold_days = trade.get('hold_days', 0)

        lines.append(f"Outcome: {outcome} - ${pnl:+.2f} ({pnl_pct:+.1f}%) in {hold_days} days")

        if trade.get('alert_text'):
            lines.append(f"Trade Thesis: {trade['alert_text'][:100]}...")

        return "\n".join(lines)

    def build_prompt(
        self,
        alert: Dict[str, Any],
        similar_trades: List[Dict[str, Any]],
        stats: Dict[str, Any]
    ) -> str:
        """
        Build prompt for Claude

        Args:
            alert: New alert
            similar_trades: Retrieved similar trades
            stats: Aggregate statistics

        Returns:
            Formatted prompt string
        """
        # Format new alert
        alert_section = f"""
## NEW ALERT TO EVALUATE

Ticker: {alert.get('ticker', 'UNKNOWN')}
Strategy: {alert.get('strategy', 'Unknown')}
Action: {alert.get('action', 'N/A')}
Strike: ${alert.get('strike_price', 'N/A')}
Expiration: {alert.get('expiration_date', 'N/A')} (DTE: {alert.get('dte', 'N/A')})
Premium: ${alert.get('premium', 'N/A')}

Current Market Conditions:
- Stock Price: ${alert.get('current_price', 'N/A')}
- VIX: {alert.get('current_vix', 'N/A')}
- IV Rank: {alert.get('iv_rank', 'N/A')}

Alert Text: {alert.get('alert_text', 'N/A')}
"""

        # Format historical trades
        historical_section = "## HISTORICAL SIMILAR TRADES\n\n"
        if similar_trades:
            historical_section += f"I found {len(similar_trades)} similar trades in your history:\n\n"
            for i, trade in enumerate(similar_trades, 1):
                historical_section += self.format_trade_for_context(trade, i) + "\n\n"
        else:
            historical_section += "No similar historical trades found.\n"

        # Format statistics
        stats_section = f"""
## AGGREGATE STATISTICS

Total Similar Trades: {stats['count']}
Win Rate: {stats['win_rate']:.1f}%
Average P&L: ${stats['avg_pnl']:.2f}
Average P&L %: {stats['avg_pnl_percent']:+.1f}%
Average Hold Time: {stats['avg_hold_days']} days
"""

        if stats['best_trade']:
            stats_section += f"Best Trade: ${stats['best_trade'].get('pnl', 0):+.2f} ({stats['best_trade'].get('pnl_percent', 0):+.1f}%)\n"
        if stats['worst_trade']:
            stats_section += f"Worst Trade: ${stats['worst_trade'].get('pnl', 0):+.2f} ({stats['worst_trade'].get('pnl_percent', 0):+.1f}%)\n"

        # Full prompt
        prompt = f"""
You are an expert options trading advisor analyzing a new trade alert from Xtrades.net.

{alert_section}

{historical_section}

{stats_section}

## YOUR TASK

Based on the historical evidence and current market conditions, provide a recommendation.

Consider:
1. Historical win rate and P&L for similar trades
2. Current market volatility (VIX)
3. Trade parameters (DTE, strike, premium)
4. Risk/reward profile
5. Any red flags or concerns

Provide your response in the following JSON format:

{{
  "recommendation": "TAKE" | "PASS" | "MONITOR",
  "confidence": 0-100,
  "reasoning": "Detailed explanation of your recommendation based on historical patterns",
  "historical_evidence": [
    "Key outcome 1",
    "Key outcome 2",
    "..."
  ],
  "risk_factors": [
    "Risk factor 1",
    "Risk factor 2",
    "..."
  ],
  "suggested_adjustments": "Optional modifications to improve probability of success"
}}

Respond ONLY with the JSON object, no additional text.
"""

        return prompt

    def parse_claude_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse Claude's JSON response

        Args:
            response_text: Raw response from Claude

        Returns:
            Parsed recommendation dictionary
        """
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                raise ValueError("No JSON found in response")

        except Exception as e:
            logger.error(f"Error parsing Claude response: {e}")
            # Return fallback response
            return {
                "recommendation": "MONITOR",
                "confidence": 50,
                "reasoning": "Unable to parse recommendation. Please review manually.",
                "historical_evidence": [],
                "risk_factors": ["Parse error occurred"],
                "suggested_adjustments": "Manual review required"
            }

    def get_recommendation(
        self,
        alert: Dict[str, Any],
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """
        Get RAG-powered recommendation for new alert (FREE via Groq/Ollama)

        Args:
            alert: New alert dictionary
            temperature: LLM temperature
            max_tokens: Maximum tokens for response

        Returns:
            Recommendation dictionary
        """
        try:
            # Search for similar trades
            similar_trades = self.search_similar_trades(alert, limit=10)

            # Re-rank and select top-5
            top_trades = self.rerank_trades(similar_trades, top_k=5)

            # Calculate statistics
            stats = self.calculate_statistics(top_trades)

            # Build prompt
            prompt = self.build_prompt(alert, top_trades, stats)

            # Call LLM (FREE via Groq/Ollama)
            logger.info("Calling FREE LLM for recommendation...")

            async def get_llm_response():
                return await self.llm.generate(
                    system="""You are an expert options trading advisor.
You analyze trade alerts by comparing them to historical outcomes.
You always provide evidence-based recommendations with clear reasoning.
You identify risks and suggest adjustments to improve probability of success.
You respond ONLY with valid JSON, no additional commentary.""",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )

            response = asyncio.run(get_llm_response())

            # Parse response
            recommendation = self.parse_claude_response(response.content)

            # Add metadata
            recommendation['similar_trades_found'] = len(similar_trades)
            recommendation['top_trades_used'] = len(top_trades)
            recommendation['statistics'] = stats
            recommendation['timestamp'] = datetime.now().isoformat()

            logger.info(
                f"Recommendation: {recommendation['recommendation']} "
                f"(Confidence: {recommendation['confidence']}%)"
            )

            return recommendation

        except Exception as e:
            logger.error(f"Error getting recommendation: {e}")
            raise


def main():
    """
    Example usage
    """
    # Initialize engine
    engine = RAGQueryEngine()

    # Example alert
    alert = {
        'ticker': 'AAPL',
        'strategy': 'CSP',
        'action': 'BTO',
        'strike_price': 170.0,
        'expiration_date': '2024-12-20',
        'dte': 30,
        'premium': 2.50,
        'current_price': 178.50,
        'current_vix': 15.2,
        'iv_rank': 45,
        'alert_text': 'BTO 2x $AAPL 12/20 $170 CSP @ $2.50 - High IV rank, bullish on tech'
    }

    # Get recommendation
    recommendation = engine.get_recommendation(alert)

    # Display results
    print("\n" + "="*80)
    print("RAG RECOMMENDATION")
    print("="*80)
    print(json.dumps(recommendation, indent=2))
    print("="*80)


if __name__ == "__main__":
    main()
