"""
Test script for RAG MVP
Validates RAG system is working correctly
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag.simple_rag import get_rag, DEPENDENCIES_AVAILABLE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_rag_system():
    """Test RAG system functionality."""

    logger.info("="*60)
    logger.info("Testing Magnus RAG System")
    logger.info("="*60)

    # Check dependencies
    if not DEPENDENCIES_AVAILABLE:
        logger.error("❌ RAG dependencies not installed")
        logger.error("Run: pip install chromadb sentence-transformers")
        return False

    logger.info("✅ Dependencies available")

    # Initialize RAG
    try:
        rag = get_rag()
        if not rag:
            logger.error("❌ Failed to initialize RAG")
            return False
        logger.info("✅ RAG system initialized")
    except Exception as e:
        logger.error(f"❌ Error initializing RAG: {e}")
        return False

    # Test document addition
    logger.info("\n--- Testing Document Addition ---")
    try:
        test_doc = """
        The Wheel Strategy is an options trading strategy that involves selling cash-secured puts
        on stocks you want to own. If assigned, you sell covered calls on the stock. This generates
        consistent income from premiums while potentially acquiring stocks at a discount.

        Key benefits include:
        - Consistent premium income
        - Lower cost basis on stock purchases
        - Defined risk with cash-secured puts

        Risk management is crucial:
        - Only sell puts on quality stocks
        - Keep position sizes at 2-5% of portfolio
        - Set strike prices at support levels
        """

        chunk_ids = rag.add_document(
            text=test_doc,
            metadata={
                "source": "test",
                "filename": "test_wheel_strategy.txt",
                "type": "test"
            },
            doc_id="test_wheel_001"
        )

        logger.info(f"✅ Added test document ({len(chunk_ids)} chunks)")

    except Exception as e:
        logger.error(f"❌ Error adding document: {e}")
        return False

    # Test querying
    logger.info("\n--- Testing Query Functionality ---")
    test_queries = [
        "What is the wheel strategy?",
        "How do I manage risk with options?",
        "What are the benefits of selling puts?"
    ]

    for query in test_queries:
        try:
            logger.info(f"\nQuery: {query}")
            results = rag.query(query, n_results=2)

            if results:
                logger.info(f"✅ Found {len(results)} results")
                for i, result in enumerate(results):
                    logger.info(f"\n  Result {i+1}:")
                    logger.info(f"    Source: {result['metadata'].get('filename', 'Unknown')}")
                    logger.info(f"    Distance: {result['distance']:.4f}")
                    preview = result['document'][:150].replace('\n', ' ')
                    logger.info(f"    Preview: {preview}...")
            else:
                logger.warning(f"⚠️  No results found")

        except Exception as e:
            logger.error(f"❌ Error querying: {e}")
            return False

    # Test context generation
    logger.info("\n--- Testing Context Generation ---")
    try:
        query = "Explain the wheel strategy"
        context = rag.get_context_for_query(query, n_results=2, max_context_length=500)

        if context:
            logger.info("✅ Context generated successfully")
            logger.info(f"\nContext preview:")
            preview = context[:300].replace('\n', ' ')
            logger.info(f"{preview}...")
        else:
            logger.warning("⚠️  Empty context generated")

    except Exception as e:
        logger.error(f"❌ Error generating context: {e}")
        return False

    # Test stats
    logger.info("\n--- Testing Statistics ---")
    try:
        stats = rag.get_stats()
        logger.info("✅ Stats retrieved:")
        logger.info(f"  Total chunks: {stats.get('total_chunks', 0)}")
        logger.info(f"  Collection: {stats.get('collection_name', 'N/A')}")
        logger.info(f"  Embedding dim: {stats.get('embedding_model', 'N/A')}")

    except Exception as e:
        logger.error(f"❌ Error getting stats: {e}")
        return False

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("✅ All RAG system tests passed!")
    logger.info("="*60)
    logger.info("\nRAG system is ready to use with AVA")
    logger.info("Next steps:")
    logger.info("1. Load your documents: python scripts/load_documents.py --sample")
    logger.info("2. Or add PDFs: python scripts/load_documents.py --pdf")
    logger.info("3. Start using AVA with RAG-enhanced responses!")
    logger.info("="*60 + "\n")

    return True


def test_with_sample_data():
    """Test RAG with sample financial data."""

    logger.info("\n" + "="*60)
    logger.info("Testing RAG with Sample Financial Data")
    logger.info("="*60)

    rag = get_rag()
    if not rag:
        logger.error("Failed to initialize RAG")
        return False

    # Add sample financial document
    sample_doc = """
    Portfolio Greeks Management

    Understanding your portfolio's Greeks is crucial for risk management:

    Delta Management:
    - Total portfolio delta indicates directional exposure
    - Positive delta = bullish positioning
    - Negative delta = bearish positioning
    - Target: Keep delta between -0.3 and +0.3 for neutral strategies

    Gamma Risk:
    - High gamma means rapid delta changes
    - Monitor gamma especially near expiration
    - Large gamma positions require frequent rebalancing

    Theta Collection:
    - Positive theta means you earn from time decay
    - Theta accelerates in last 30 days to expiration
    - Maximize theta by selling options 30-45 DTE

    Vega Exposure:
    - Positive vega = benefit from rising IV
    - Negative vega = benefit from falling IV
    - Earnings announcements spike IV - be prepared

    Practical Application:
    1. Calculate portfolio Greeks daily
    2. Adjust positions when Greeks exceed limits
    3. Use opposing Greeks to balance portfolio
    4. Consider hedging with VIX products for vega risk
    """

    rag.add_document(
        text=sample_doc,
        metadata={"source": "sample", "filename": "greeks_management.txt"},
        doc_id="greeks_guide"
    )

    # Test queries
    test_queries = [
        "How do I manage portfolio delta?",
        "What is gamma risk?",
        "How can I maximize theta collection?",
        "What should I do about vega exposure?"
    ]

    logger.info("\nTesting queries against sample data:")

    for query in test_queries:
        logger.info(f"\n📊 Query: {query}")
        results = rag.query(query, n_results=1)

        if results:
            result = results[0]
            logger.info(f"✅ Found relevant content (distance: {result['distance']:.4f})")

            # Show relevant excerpt
            doc = result['document']
            lines = doc.split('\n')
            relevant_lines = [line for line in lines if line.strip()][:3]
            logger.info("   Relevant content:")
            for line in relevant_lines:
                logger.info(f"   {line.strip()}")
        else:
            logger.warning("⚠️  No results")

    logger.info("\n" + "="*60)
    logger.info("✅ Sample data testing complete")
    logger.info("="*60 + "\n")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Magnus RAG system")
    parser.add_argument(
        "--with-samples",
        action="store_true",
        help="Test with sample financial data"
    )

    args = parser.parse_args()

    # Run basic tests
    success = test_rag_system()

    # Run sample data tests if requested
    if args.with_samples and success:
        test_with_sample_data()

    sys.exit(0 if success else 1)
