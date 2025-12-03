"""
Document Loader for Magnus RAG System
======================================

Loads documents from various formats into the knowledge base using UnifiedRAG.

Usage:
    python scripts/load_documents.py                    # Load all documents
    python scripts/load_documents.py --clear            # Clear and reload
    python scripts/load_documents.py --pdf              # Include PDF files
    python scripts/load_documents.py --sample           # Load sample data only

Categories automatically detected from directory structure:
    data/documents/options/    -> options_education
    data/documents/risk/       -> risk_management
    data/documents/technical/  -> technical_analysis
    data/documents/platform/   -> platform_docs
    data/documents/sports/     -> sports_betting
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the new UnifiedRAG interface
try:
    from src.rag import get_unified_rag, UnifiedRAG
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"Warning: RAG dependencies not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_text_files(directory: str = "data/documents", recursive: bool = True):
    """
    Load all text files from a directory.

    Args:
        directory: Directory containing documents
        recursive: Whether to search recursively
    """
    if not DEPENDENCIES_AVAILABLE:
        logger.error("RAG dependencies not installed. Run: pip install chromadb sentence-transformers")
        return

    logger.info(f"Loading documents from: {directory}")

    # Get RAG instance
    rag = get_unified_rag()
    if not rag:
        logger.error("Failed to initialize RAG system")
        return

    # Check if directory exists
    if not os.path.exists(directory):
        logger.warning(f"Directory not found: {directory}")
        logger.info(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Please add your documents to {directory} and run this script again")
        return

    # Load text and markdown files
    logger.info("Loading .txt and .md files...")
    all_results = rag.add_documents_from_directory(
        directory=directory,
        file_patterns=["*.txt", "*.md"],
        recursive=recursive
    )

    # Report results
    total_files = len(all_results)
    total_chunks = sum(all_results.values())

    logger.info(f"\n{'='*60}")
    logger.info(f"Document Loading Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Total files loaded: {total_files}")
    logger.info(f"Total chunks created: {total_chunks}")

    if all_results:
        logger.info(f"\nLoaded files:")
        for filename, chunk_count in all_results.items():
            logger.info(f"  - {filename}: {chunk_count} chunks")

    # Get stats
    stats = rag.get_stats()
    logger.info(f"\nKnowledge base stats:")
    logger.info(f"  Total chunks: {stats.get('total_chunks', 0)}")
    logger.info(f"  Collection: {stats.get('collection_name', 'N/A')}")

    logger.info(f"\n{'='*60}")
    logger.info("RAG system is ready to use!")
    logger.info(f"{'='*60}\n")


def load_pdf_files(directory: str = "data/documents", recursive: bool = True):
    """
    Load PDF files (requires pypdf).

    Args:
        directory: Directory containing PDFs
        recursive: Whether to search recursively
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf not installed. Run: pip install pypdf")
        return

    if not DEPENDENCIES_AVAILABLE:
        logger.error("RAG dependencies not installed. Run: pip install chromadb sentence-transformers")
        return

    logger.info(f"Loading PDF files from: {directory}")

    # Get RAG instance
    rag = get_unified_rag()
    if not rag:
        logger.error("Failed to initialize RAG system")
        return

    # Find PDF files
    path = Path(directory)
    if not path.exists():
        logger.error(f"Directory not found: {directory}")
        return

    if recursive:
        pdf_files = list(path.rglob("*.pdf"))
    else:
        pdf_files = list(path.glob("*.pdf"))

    logger.info(f"Found {len(pdf_files)} PDF files")

    # Process each PDF
    results = {}
    for pdf_path in pdf_files:
        try:
            logger.info(f"Processing: {pdf_path.name}")

            # Read PDF
            reader = PdfReader(str(pdf_path))
            text_parts = []

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{text}")

            full_text = "\n\n".join(text_parts)

            # Add to RAG
            metadata = {
                "source": str(pdf_path),
                "filename": pdf_path.name,
                "file_type": ".pdf",
                "pages": len(reader.pages)
            }

            chunk_ids = rag.add_document(
                text=full_text,
                metadata=metadata,
                doc_id=pdf_path.stem
            )

            results[str(pdf_path)] = chunk_ids
            logger.info(f"  Added {pdf_path.name}: {len(chunk_ids)} chunks from {len(reader.pages)} pages")

        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {e}")

    # Report results
    total_files = len(results)
    total_chunks = sum(len(chunks) for chunks in results.values())

    logger.info(f"\n{'='*60}")
    logger.info(f"PDF Loading Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Total PDFs loaded: {total_files}")
    logger.info(f"Total chunks created: {total_chunks}")

    if results:
        logger.info(f"\nLoaded PDFs:")
        for file_path, chunks in results.items():
            logger.info(f"  - {Path(file_path).name}: {len(chunks)} chunks")


def load_sample_data():
    """Load sample financial documents for testing."""
    logger.info("Loading sample financial data...")

    rag = get_unified_rag()
    if not rag:
        logger.error("Failed to initialize RAG system")
        return

    # Sample documents about financial concepts
    samples = [
        {
            "title": "Wheel Strategy Guide",
            "content": """
The Wheel Strategy is a popular options trading strategy that involves:

1. Selling cash-secured puts on stocks you want to own
2. If assigned, sell covered calls on the stock
3. If called away, repeat the process

Key Benefits:
- Generate consistent income from premiums
- Acquire stocks at a discount
- Lower cost basis over time

Risk Management:
- Only sell puts on quality stocks you're willing to own
- Keep position sizes manageable (2-5% of portfolio)
- Set strike prices at support levels
- Maintain adequate cash reserves

Best Practices:
- Target 30-45 DTE (Days To Expiration)
- Aim for 0.30 delta strikes
- Roll positions when profitable (50-75% max profit)
- Avoid earnings announcements
"""
        },
        {
            "title": "Options Greeks Explained",
            "content": """
Understanding Options Greeks:

Delta: Rate of price change relative to underlying
- Call delta: 0 to 1 (increases as ITM)
- Put delta: -1 to 0 (decreases as ITM)
- Portfolio delta measures directional exposure

Gamma: Rate of delta change
- Highest near ATM options
- Increases as expiration approaches
- High gamma = more position adjustments needed

Theta: Time decay per day
- Always negative for long options
- Positive for short options
- Accelerates in final 30 days

Vega: Price change per 1% IV move
- Long options = positive vega
- Short options = negative vega
- Higher for longer-dated options

Risk Management with Greeks:
- Monitor portfolio delta for directional risk
- High gamma requires more frequent adjustments
- Theta decay works for sellers, against buyers
- Vega risk increases during earnings/events
"""
        },
        {
            "title": "Technical Analysis Fundamentals",
            "content": """
Key Technical Analysis Concepts:

Support and Resistance:
- Support: Price level where buying pressure increases
- Resistance: Price level where selling pressure increases
- Break and retest patterns confirm strength

Trend Analysis:
- Uptrend: Higher highs and higher lows
- Downtrend: Lower highs and lower lows
- Sideways: Consolidation phase

Volume Analysis:
- Volume confirms price moves
- Low volume breakouts often fail
- High volume at support/resistance = significant

Common Indicators:
- RSI (14): Overbought >70, Oversold <30
- MACD: Momentum and trend indicator
- Moving Averages: 20, 50, 200 day common
- Bollinger Bands: Volatility and price extremes

Entry/Exit Strategies:
- Enter on pullbacks to support in uptrend
- Wait for confirmation on breakouts
- Use stop losses below support
- Take profits at resistance levels
"""
        },
        {
            "title": "Risk Management Principles",
            "content": """
Essential Risk Management Rules:

Position Sizing:
- Risk no more than 1-2% of portfolio per trade
- Size based on stop loss distance
- Larger accounts can use smaller percentages
- Never risk more than you can afford to lose

Portfolio Diversification:
- Spread risk across multiple positions
- Avoid concentration in single sector
- Mix of bullish, bearish, and neutral strategies
- Consider correlation between positions

Stop Loss Strategy:
- Always define exit before entry
- Place stops below technical support
- Avoid arbitrary percentage stops
- Don't move stops against position

Profit Taking:
- Define profit targets before entry
- Scale out of winners
- Let winners run with trailing stops
- Don't let big winners become losers

Emotional Control:
- Stick to your trading plan
- Don't revenge trade after losses
- Take breaks after big wins/losses
- Keep trading journal for review
"""
        }
    ]

    # Add sample documents
    for sample in samples:
        try:
            metadata = {
                "source": "sample_data",
                "filename": f"{sample['title']}.txt",
                "file_type": ".txt",
                "type": "educational"
            }

            chunk_ids = rag.add_document(
                text=sample['content'],
                metadata=metadata,
                doc_id=sample['title'].lower().replace(' ', '_')
            )

            logger.info(f"Added: {sample['title']} ({len(chunk_ids)} chunks)")

        except Exception as e:
            logger.error(f"Error adding {sample['title']}: {e}")

    # Get stats
    stats = rag.get_stats()
    logger.info(f"\nSample data loaded successfully!")
    logger.info(f"Total chunks in knowledge base: {stats.get('total_chunks', 0)}")


def clear_knowledge_base():
    """Clear all documents from knowledge base."""
    logger.warning("Clearing knowledge base...")

    rag = get_unified_rag()
    if not rag:
        logger.error("Failed to initialize RAG system")
        return

    rag.clear_collection()
    logger.info("Knowledge base cleared")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load documents into Magnus RAG system")
    parser.add_argument(
        "--directory",
        type=str,
        default="data/documents",
        help="Directory containing documents (default: data/documents)"
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Load PDF files (requires pypdf)"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Load sample financial documents"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear knowledge base before loading"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Search directories recursively (default: True)"
    )

    args = parser.parse_args()

    # Clear if requested
    if args.clear:
        clear_knowledge_base()

    # Load sample data if requested
    if args.sample:
        load_sample_data()

    # Load text/markdown files
    if not args.pdf:
        load_text_files(directory=args.directory, recursive=args.recursive)

    # Load PDFs if requested
    if args.pdf:
        load_pdf_files(directory=args.directory, recursive=args.recursive)

    logger.info("\nDone!")
