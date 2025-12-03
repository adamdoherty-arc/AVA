"""
RAG Embedding Pipeline - Convert trades to semantic vectors

This module handles:
1. Loading historical trades from PostgreSQL
2. Enriching with market data
3. Generating embeddings using Hugging Face
4. Storing in Qdrant vector database

Author: Magnus Wheel Strategy Dashboard
Created: 2025-11-06
"""

import os
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import yfinance as yf
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range
)
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class TradeEmbeddingPipeline:
    """
    Pipeline for converting trades to vector embeddings
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None
    ):
        """
        Initialize the embedding pipeline

        Args:
            embedding_model: Hugging Face model name
            qdrant_url: Qdrant cluster URL (default: from env)
            qdrant_api_key: Qdrant API key (default: from env)
        """
        # Database configuration
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('DB_NAME', 'magnus')
        }

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")

        # Initialize Qdrant client
        self.qdrant_url = qdrant_url or os.getenv('QDRANT_URL')
        self.qdrant_api_key = qdrant_api_key or os.getenv('QDRANT_API_KEY')

        if not self.qdrant_api_key:
            raise ValueError("QDRANT_API_KEY must be set in environment")

        logger.info(f"Connecting to Qdrant at {self.qdrant_url or 'default URL'}")
        self.qdrant = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )

        self.collection_name = "options_trades"

        # Market data cache
        self.market_data_cache = {}

    def get_db_connection(self) -> None:
        """Get PostgreSQL connection"""
        return psycopg2.connect(**self.db_config)

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create Qdrant collection for options trades

        Args:
            recreate: If True, delete existing collection and recreate
        """
        try:
            # Check if collection exists
            collections = self.qdrant.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)

            if collection_exists and recreate:
                logger.warning(f"Deleting existing collection: {self.collection_name}")
                self.qdrant.delete_collection(self.collection_name)
                collection_exists = False

            if not collection_exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info("Collection created successfully")

                # Create payload indexes for faster filtering
                self._create_indexes()
            else:
                logger.info(f"Collection {self.collection_name} already exists")

        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def _create_indexes(self) -> None:
        """Create payload indexes for optimized filtering"""
        indexes = [
            ("ticker", "keyword"),
            ("strategy", "keyword"),
            ("status", "keyword"),
            ("win", "keyword"),
            ("profile_username", "keyword"),
            ("sector", "keyword"),
            ("spy_trend", "keyword"),
            ("dte", "integer"),
            ("vix_at_entry", "float"),
            ("iv_rank", "integer"),
            ("entry_date", "datetime")
        ]

        for field, field_type in indexes:
            try:
                self.qdrant.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=field_type
                )
                logger.info(f"Created index on {field} ({field_type})")
            except Exception as e:
                logger.warning(f"Could not create index on {field}: {e}")

    def load_trades_from_db(
        self,
        status: str = 'closed',
        limit: Optional[int] = None,
        min_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Load trades from PostgreSQL database

        Args:
            status: Trade status filter ('closed', 'open', None for all)
            limit: Maximum number of trades to load
            min_date: Only load trades after this date

        Returns:
            List of trade dictionaries
        """
        conn = self.get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            query = """
                SELECT
                    t.id, t.ticker, t.strategy, t.action,
                    t.entry_price, t.exit_price, t.entry_date, t.exit_date,
                    t.quantity, t.pnl, t.pnl_percent, t.status,
                    t.strike_price, t.expiration_date, t.alert_text,
                    t.alert_timestamp,
                    p.username as profile_username
                FROM xtrades_trades t
                JOIN xtrades_profiles p ON t.profile_id = p.id
                WHERE 1=1
            """

            params = []

            if status:
                query += " AND t.status = %s"
                params.append(status)

            if min_date:
                query += " AND t.entry_date >= %s"
                params.append(min_date)

            query += " ORDER BY t.entry_date DESC"

            if limit:
                query += f" LIMIT {limit}"

            logger.info(f"Loading trades from database (status={status}, limit={limit})")
            cur.execute(query, params)

            trades = [dict(trade) for trade in cur.fetchall()]
            logger.info(f"Loaded {len(trades)} trades")

            return trades

        except Exception as e:
            logger.error(f"Error loading trades: {e}")
            return []
        finally:
            cur.close()
            conn.close()

    def enrich_trade_with_market_data(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich trade with market data (sector, VIX, etc.)

        Args:
            trade: Trade dictionary

        Returns:
            Enriched trade dictionary
        """
        ticker = trade.get('ticker')
        entry_date = trade.get('entry_date')

        if not ticker or not entry_date:
            return trade

        try:
            # Get stock info
            stock_info = self._get_stock_info(ticker)

            trade['sector'] = stock_info.get('sector', 'Unknown')
            trade['market_cap'] = self._categorize_market_cap(stock_info.get('marketCap', 0))

            # Get VIX data (approximate - use cached value or fetch)
            trade['vix_at_entry'] = self._get_vix_on_date(entry_date)

            # Get SPY trend
            trade['spy_trend'] = self._get_spy_trend(entry_date)

            # Calculate DTE if expiration date available
            if trade.get('expiration_date') and entry_date:
                dte = (trade['expiration_date'] - entry_date).days
                trade['dte'] = max(0, dte)

            # Calculate hold days for closed trades
            if trade.get('exit_date') and entry_date:
                trade['hold_days'] = (trade['exit_date'] - entry_date).days

            # Determine if trade was a win
            trade['win'] = trade.get('pnl', 0) > 0

            # Get stock price at entry (approximate)
            trade['stock_price_at_entry'] = self._get_stock_price_on_date(
                ticker, entry_date
            )

            if trade.get('exit_date'):
                trade['stock_price_at_exit'] = self._get_stock_price_on_date(
                    ticker, trade['exit_date']
                )

            # Placeholder for Greeks (would need options data API)
            trade['delta'] = None
            trade['theta'] = None
            trade['iv_rank'] = None  # Would need historical IV data

        except Exception as e:
            logger.warning(f"Could not enrich trade {trade.get('id')}: {e}")

        return trade

    def _get_stock_info(self, ticker: str) -> Dict[str, Any]:
        """Get stock information from cache or Yahoo Finance"""
        if ticker in self.market_data_cache:
            return self.market_data_cache[ticker]

        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            self.market_data_cache[ticker] = info
            return info
        except Exception as e:
            logger.warning(f"Could not fetch info for {ticker}: {e}")
            return {}

    def _categorize_market_cap(self, market_cap: float) -> str:
        """Categorize market cap into Large/Mid/Small"""
        if market_cap > 10e9:  # > $10B
            return "Large"
        elif market_cap > 2e9:  # $2B - $10B
            return "Mid"
        else:
            return "Small"

    def _get_vix_on_date(self, date: datetime) -> float:
        """Get VIX level on specific date (approximate)"""
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(start=date - timedelta(days=7), end=date + timedelta(days=1))
            if not hist.empty:
                return round(hist['Close'].iloc[-1], 2)
        except:
            pass
        return 15.0  # Default VIX if unavailable

    def _get_spy_trend(self, date: datetime) -> str:
        """Determine SPY trend (Bullish/Bearish/Neutral)"""
        try:
            spy = yf.Ticker("SPY")
            hist = spy.history(start=date - timedelta(days=30), end=date + timedelta(days=1))
            if len(hist) >= 2:
                sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
                current_price = hist['Close'].iloc[-1]

                if current_price > sma_20 * 1.02:
                    return "Bullish"
                elif current_price < sma_20 * 0.98:
                    return "Bearish"
        except:
            pass
        return "Neutral"

    def _get_stock_price_on_date(self, ticker: str, date: datetime) -> Optional[float]:
        """Get stock price on specific date"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=date - timedelta(days=7), end=date + timedelta(days=1))
            if not hist.empty:
                return round(hist['Close'].iloc[-1], 2)
        except:
            pass
        return None

    def format_trade_for_embedding(self, trade: Dict[str, Any]) -> str:
        """
        Format trade into text for embedding

        Args:
            trade: Enriched trade dictionary

        Returns:
            Formatted text string
        """
        # Create comprehensive text representation
        text_parts = []

        # Strategy and ticker
        text_parts.append(f"Strategy: {trade.get('strategy', 'Unknown')}")
        text_parts.append(f"Ticker: {trade.get('ticker', 'UNKNOWN')}")
        if trade.get('sector'):
            text_parts.append(f"Sector: {trade['sector']}")

        # Action and position details
        text_parts.append(f"Action: {trade.get('action', 'UNKNOWN')}")
        if trade.get('strike_price'):
            text_parts.append(f"Strike: ${trade['strike_price']:.2f}")
        if trade.get('dte'):
            text_parts.append(f"Days to Expiration: {trade['dte']}")

        # Premium and pricing
        if trade.get('entry_price'):
            text_parts.append(f"Entry Price: ${trade['entry_price']:.2f}")
        if trade.get('premium'):
            text_parts.append(f"Premium: ${trade['premium']:.2f}")

        # Market conditions
        if trade.get('vix_at_entry'):
            text_parts.append(f"VIX: {trade['vix_at_entry']:.1f}")
        if trade.get('spy_trend'):
            text_parts.append(f"Market Trend: {trade['spy_trend']}")
        if trade.get('iv_rank'):
            text_parts.append(f"IV Rank: {trade['iv_rank']}")

        # Trade outcome (for closed trades)
        if trade.get('status') == 'closed':
            outcome = "Win" if trade.get('win') else "Loss"
            text_parts.append(f"Outcome: {outcome}")
            if trade.get('pnl'):
                text_parts.append(f"P&L: ${trade['pnl']:.2f}")
            if trade.get('pnl_percent'):
                text_parts.append(f"P&L Percent: {trade['pnl_percent']:+.1f}%")
            if trade.get('hold_days'):
                text_parts.append(f"Hold Time: {trade['hold_days']} days")

        # Alert text (contains trade thesis)
        if trade.get('alert_text'):
            text_parts.append(f"Alert: {trade['alert_text']}")

        return "\n".join(text_parts)

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text

        Args:
            text: Input text

        Returns:
            Embedding vector (list of floats)
        """
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def create_point(self, trade: Dict[str, Any]) -> PointStruct:
        """
        Create Qdrant point from trade

        Args:
            trade: Enriched trade dictionary

        Returns:
            PointStruct for insertion
        """
        # Format text for embedding
        embedded_text = self.format_trade_for_embedding(trade)

        # Generate embedding
        embedding = self.generate_embedding(embedded_text)

        # Create payload (metadata)
        payload = {
            'trade_id': trade['id'],
            'ticker': trade.get('ticker', '').upper(),
            'strategy': trade.get('strategy'),
            'action': trade.get('action'),
            'strike_price': float(trade['strike_price']) if trade.get('strike_price') else None,
            'expiration_date': trade['expiration_date'].isoformat() if trade.get('expiration_date') else None,
            'dte': trade.get('dte'),
            'premium': float(trade['premium']) if trade.get('premium') else None,
            'entry_price': float(trade['entry_price']) if trade.get('entry_price') else None,
            'exit_price': float(trade['exit_price']) if trade.get('exit_price') else None,
            'quantity': trade.get('quantity', 1),
            'pnl': float(trade['pnl']) if trade.get('pnl') else None,
            'pnl_percent': float(trade['pnl_percent']) if trade.get('pnl_percent') else None,
            'status': trade.get('status'),
            'win': trade.get('win', False),
            'entry_date': trade['entry_date'].isoformat() if trade.get('entry_date') else None,
            'exit_date': trade['exit_date'].isoformat() if trade.get('exit_date') else None,
            'hold_days': trade.get('hold_days'),
            'profile_username': trade.get('profile_username'),
            'sector': trade.get('sector'),
            'market_cap': trade.get('market_cap'),
            'vix_at_entry': trade.get('vix_at_entry'),
            'spy_trend': trade.get('spy_trend'),
            'iv_rank': trade.get('iv_rank'),
            'delta': trade.get('delta'),
            'theta': trade.get('theta'),
            'stock_price_at_entry': trade.get('stock_price_at_entry'),
            'stock_price_at_exit': trade.get('stock_price_at_exit'),
            'alert_text': trade.get('alert_text'),
            'embedded_text': embedded_text,
            'success_weight': 1.0,
            'times_referenced': 0,
            'avg_recommendation_accuracy': 0.0,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }

        # Create point
        point_id = f"trade_{trade['id']}"

        return PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        )

    def index_trades(
        self,
        trades: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """
        Index trades into Qdrant

        Args:
            trades: List of trade dictionaries
            batch_size: Number of trades to batch together

        Returns:
            Number of trades indexed
        """
        logger.info(f"Indexing {len(trades)} trades...")

        indexed_count = 0
        batch = []

        for i, trade in enumerate(trades):
            try:
                # Enrich trade
                enriched_trade = self.enrich_trade_with_market_data(trade)

                # Create point
                point = self.create_point(enriched_trade)
                batch.append(point)

                # Insert batch
                if len(batch) >= batch_size or i == len(trades) - 1:
                    self.qdrant.upsert(
                        collection_name=self.collection_name,
                        points=batch
                    )
                    indexed_count += len(batch)
                    logger.info(f"Indexed {indexed_count}/{len(trades)} trades")
                    batch = []

            except Exception as e:
                logger.error(f"Error indexing trade {trade.get('id')}: {e}")
                continue

        logger.info(f"Successfully indexed {indexed_count} trades")
        return indexed_count

    def backfill_from_database(
        self,
        recreate_collection: bool = False,
        limit: Optional[int] = None
    ) -> int:
        """
        Backfill Qdrant with historical trades from PostgreSQL

        Args:
            recreate_collection: If True, recreate collection from scratch
            limit: Maximum number of trades to index

        Returns:
            Number of trades indexed
        """
        logger.info("Starting backfill process...")

        # Create collection
        self.create_collection(recreate=recreate_collection)

        # Load trades from database
        trades = self.load_trades_from_db(status='closed', limit=limit)

        if not trades:
            logger.warning("No trades to index")
            return 0

        # Index trades
        indexed_count = self.index_trades(trades)

        logger.info(f"Backfill complete. Indexed {indexed_count} trades.")
        return indexed_count


def main():
    """
    Example usage: Backfill Qdrant with historical trades
    """
    pipeline = TradeEmbeddingPipeline()

    # Backfill with all closed trades
    indexed_count = pipeline.backfill_from_database(
        recreate_collection=True,  # Start fresh
        limit=None  # Index all trades
    )

    print(f"\nBackfill complete!")
    print(f"Indexed {indexed_count} trades")


if __name__ == "__main__":
    main()
