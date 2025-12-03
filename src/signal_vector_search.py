"""
Signal Vector Search using ChromaDB
Enables semantic similarity search for finding similar past trades
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import psycopg2
import psycopg2.extras
import os
from datetime import datetime


class SignalVectorSearch:
    """Vector search for trading signals using ChromaDB"""

    def __init__(self) -> None:
        self.db_password = os.getenv('DB_PASSWORD')

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )

        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="trading_signals",
            metadata={"description": "Discord trading signals with outcomes"}
        )

    def get_connection(self) -> None:
        """Get PostgreSQL connection"""
        return psycopg2.connect(
            host='localhost',
            port='5432',
            database='magnus',
            user='postgres',
            password=self.db_password
        )

    def create_signal_embedding_text(self, signal: Dict) -> str:
        """Create text representation of signal for embedding"""
        parts = []

        # Ticker and setup
        if signal.get('primary_ticker'):
            parts.append(f"Ticker: {signal['primary_ticker']}")

        if signal.get('setup_type'):
            setup = signal['setup_type'].replace('_', ' ').title()
            parts.append(f"Setup: {setup}")

        # Sentiment
        if signal.get('sentiment'):
            parts.append(f"Sentiment: {signal['sentiment'].title()}")

        # Price levels
        if signal.get('entry'):
            parts.append(f"Entry: ${signal['entry']}")
        if signal.get('target'):
            parts.append(f"Target: ${signal['target']}")
        if signal.get('stop_loss'):
            parts.append(f"Stop: ${signal['stop_loss']}")

        # Option info
        if signal.get('option_strike'):
            parts.append(f"Strike: ${signal['option_strike']}")
        if signal.get('option_type'):
            parts.append(f"Type: {signal['option_type']}")

        # Original content (truncated)
        if signal.get('content'):
            content = signal['content'][:200]
            parts.append(f"Message: {content}")

        return " | ".join(parts)

    def index_all_signals(self) -> None:
        """Index all signals from database into ChromaDB"""
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Get all signals with outcomes
        cur.execute("""
            SELECT
                s.id,
                s.primary_ticker,
                s.setup_type,
                s.sentiment,
                s.entry,
                s.target,
                s.stop_loss,
                s.option_strike,
                s.option_type,
                s.content,
                s.author,
                s.timestamp,
                o.outcome,
                o.pnl_percent,
                o.pnl_dollars
            FROM discord_trading_signals s
            LEFT JOIN signal_outcomes o ON s.id = o.signal_id
        """)

        signals = cur.fetchall()
        cur.close()
        conn.close()

        if not signals:
            print("No signals to index")
            return

        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []

        for signal in signals:
            # Create embedding text
            doc_text = self.create_signal_embedding_text(signal)
            documents.append(doc_text)

            # Metadata
            metadata = {
                'signal_id': signal['id'],
                'ticker': signal['primary_ticker'] or 'unknown',
                'setup_type': signal['setup_type'] or 'general',
                'sentiment': signal['sentiment'] or 'neutral',
                'author': signal['author'] or 'unknown',
                'outcome': signal['outcome'] or 'pending',
                'timestamp': signal['timestamp'].isoformat() if signal['timestamp'] else None,
            }

            # Add outcome data if available
            if signal.get('pnl_percent'):
                metadata['pnl_percent'] = float(signal['pnl_percent'])
            if signal.get('pnl_dollars'):
                metadata['pnl_dollars'] = float(signal['pnl_dollars'])

            metadatas.append(metadata)
            ids.append(f"signal_{signal['id']}")

        # Add to ChromaDB
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        print(f"Indexed {len(signals)} signals into ChromaDB")

    def find_similar_signals(
        self,
        signal_id: int,
        n_results: int = 5,
        only_winners: bool = False
    ) -> List[Dict]:
        """Find similar past signals"""
        # Get the signal to search for
        conn = self.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("""
            SELECT
                id, primary_ticker, setup_type, sentiment, entry, target,
                stop_loss, option_strike, option_type, content
            FROM discord_trading_signals
            WHERE id = %s
        """, (signal_id,))

        signal = cur.fetchone()
        cur.close()
        conn.close()

        if not signal:
            return []

        # Create query text
        query_text = self.create_signal_embedding_text(signal)

        # Build where clause
        where_clause = {}
        if only_winners:
            where_clause['outcome'] = 'win'

        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results + 1,  # +1 because it might return itself
            where=where_clause if where_clause else None
        )

        similar_signals = []
        if results and results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                # Skip if it's the same signal
                if doc_id == f"signal_{signal_id}":
                    continue

                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i] if 'distances' in results else None

                similar_signals.append({
                    'signal_id': metadata['signal_id'],
                    'ticker': metadata['ticker'],
                    'setup_type': metadata['setup_type'],
                    'sentiment': metadata['sentiment'],
                    'author': metadata['author'],
                    'outcome': metadata['outcome'],
                    'pnl_percent': metadata.get('pnl_percent'),
                    'similarity_score': 100 - (distance * 100) if distance else None,
                    'document': results['documents'][0][i]
                })

        return similar_signals[:n_results]

    def search_by_criteria(
        self,
        query: str,
        ticker: Optional[str] = None,
        setup_type: Optional[str] = None,
        outcome: Optional[str] = None,
        n_results: int = 10
    ) -> List[Dict]:
        """Search signals by natural language query with filters"""
        where_clause = {}

        if ticker:
            where_clause['ticker'] = ticker
        if setup_type:
            where_clause['setup_type'] = setup_type
        if outcome:
            where_clause['outcome'] = outcome

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause if where_clause else None
        )

        search_results = []
        if results and results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i] if 'distances' in results else None

                search_results.append({
                    'signal_id': metadata['signal_id'],
                    'ticker': metadata['ticker'],
                    'setup_type': metadata['setup_type'],
                    'sentiment': metadata['sentiment'],
                    'author': metadata['author'],
                    'outcome': metadata['outcome'],
                    'pnl_percent': metadata.get('pnl_percent'),
                    'relevance_score': 100 - (distance * 100) if distance else None,
                    'document': results['documents'][0][i]
                })

        return search_results

    def calculate_similarity_score_for_signal(self, signal_id: int) -> float:
        """Calculate similarity to successful past trades"""
        similar = self.find_similar_signals(signal_id, n_results=10, only_winners=True)

        if not similar:
            return 50.0  # Default neutral score

        # Average similarity score to winning trades
        scores = [s['similarity_score'] for s in similar if s.get('similarity_score')]

        if scores:
            return sum(scores) / len(scores)
        else:
            return 50.0


if __name__ == "__main__":
    search = SignalVectorSearch()
    print("Indexing all signals into ChromaDB...")
    search.index_all_signals()
    print("Vector search system ready!")
