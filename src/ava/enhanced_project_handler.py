"""
Enhanced AVA Project Knowledge Handler
======================================

Extends AVA's NLP handler to answer questions about Magnus project itself.

Features:
- Understands Magnus features and capabilities
- Explains how to use different components
- Provides code references and documentation
- Integrates with RAG for contextual answers

Author: Claude Code
Date: 2025-11-11
"""

import logging
from typing import Dict, Optional, List
from src.services.llm_service import LLMService

try:
    from src.rag.rag_service import ProductionRAGService
except ImportError:
    ProductionRAGService = None

logger = logging.getLogger(__name__)


class EnhancedProjectHandler:
    """
    Handles questions about Magnus project itself

    Augments AVA with deep knowledge of Magnus features, code,
    architecture, and usage patterns.
    """

    def __init__(self) -> None:
        """Initialize handler with RAG and LLM services"""
        self.llm_service = LLMService()

        if ProductionRAGService:
            try:
                self.rag = ProductionRAGService()
                logger.info("[OK] RAG service initialized for project knowledge")
            except Exception as e:
                logger.warning(f"Could not initialize RAG: {e}")
                self.rag = None
        else:
            self.rag = None
            logger.warning("RAG service not available")

    def is_project_question(self, user_text: str) -> bool:
        """Detect if user is asking about Magnus project itself"""
        project_keywords = [
            'magnus', 'features', 'capabilities', 'how to',
            'what can', 'where is', 'show me', 'explain',
            'dashboard', 'integrate', 'use the', 'access',
            'database', 'api', 'function', 'code', 'file',
            'page', 'module', 'system', 'architecture'
        ]

        text_lower = user_text.lower()
        return any(keyword in text_lower for keyword in project_keywords)

    def answer_project_question(self, user_text: str) -> Dict:
        """
        Answer questions about Magnus project using RAG + LLM

        Args:
            user_text: User's question about Magnus

        Returns:
            Dictionary with answer and source references
        """
        logger.info(f"Processing project question: {user_text}")

        try:
            # Step 1: Get relevant context from RAG
            context = self._get_rag_context(user_text)

            # Step 2: Generate answer using LLM + context
            answer = self._generate_answer(user_text, context)

            return {
                'success': True,
                'answer': answer['text'],
                'sources': context.get('sources', []),
                'model_used': f"{answer['provider']}/{answer['model']}",
                'cost': answer.get('cost', 0.0)
            }

        except Exception as e:
            logger.error(f"Error answering project question: {e}")
            return {
                'success': False,
                'answer': self._get_fallback_answer(user_text),
                'sources': [],
                'error': str(e)
            }

    def _get_rag_context(self, question: str) -> Dict:
        """Retrieve relevant Magnus project context from RAG"""
        if not self.rag:
            return {'sources': [], 'context_text': ''}

        try:
            # Query RAG for relevant Magnus knowledge
            results = self.rag.query(
                question=question,
                n_results=5,
                filter_metadata={'source': 'magnus_project_knowledge'}
            )

            if not results or not results.get('documents'):
                logger.warning("No RAG results found for project question")
                return {'sources': [], 'context_text': ''}

            # Build context from results
            documents = results['documents'][0] if results['documents'] else []
            metadata = results['metadatas'][0] if results['metadatas'] else []

            context_parts = []
            sources = []

            for i, doc in enumerate(documents[:5]):
                context_parts.append(f"\n=== Context {i+1} ===\n{doc}")
                if i < len(metadata):
                    sources.append({
                        'section': metadata[i].get('type', 'unknown'),
                        'relevance': results.get('distances', [[]])[0][i] if results.get('distances') else None
                    })

            return {
                'sources': sources,
                'context_text': '\n'.join(context_parts)
            }

        except Exception as e:
            logger.error(f"Error querying RAG: {e}")
            return {'sources': [], 'context_text': ''}

    def _generate_answer(self, question: str, context: Dict) -> Dict:
        """Generate answer using LLM with Magnus context"""
        context_text = context.get('context_text', '')

        prompt = f"""You are AVA, the Magnus Trading Dashboard financial assistant.

Answer this question about the Magnus project using the provided context.

**Question:** {question}

**Magnus Project Context:**
{context_text if context_text else "No specific context available. Use general knowledge about Magnus trading dashboard."}

**Instructions:**
1. Provide a clear, concise answer
2. Reference specific features, files, or modules when relevant
3. Include practical usage examples if applicable
4. Mention file paths using the pattern "file_path:line" when referring to code
5. If context is insufficient, acknowledge what you don't know

**Answer:**"""

        try:
            response = self.llm_service.generate(
                prompt=prompt,
                provider=None,  # Auto-select best provider
                max_tokens=800,
                temperature=0.3,  # Balanced creativity vs. accuracy
                use_cache=True
            )

            return response

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'text': self._get_fallback_answer(question),
                'provider': 'fallback',
                'model': 'none',
                'cost': 0.0
            }

    def _get_fallback_answer(self, question: str) -> str:
        """Provide basic fallback answer without LLM"""
        if 'features' in question.lower() or 'capabilities' in question.lower():
            return """Magnus is a comprehensive options trading dashboard with these key features:

1. **Dashboard** - Portfolio overview and metrics
2. **Positions** - Live tracking from Robinhood
3. **Opportunities** - CSP opportunity finder
4. **Premium Scanner** - Options flow analysis
5. **AI Options Agent** - Position recommendations
6. **Comprehensive Strategy** - Multi-factor analysis
7. **Database Scan** - Market-wide scanning
8. **XTrades Integration** - Professional signals
9. **Earnings Calendar** - Event tracking
10. **Prediction Markets** - Kalshi integration
11. **AVA Assistant** - Natural language interface

For more details, ask about a specific feature!"""

        elif 'how to' in question.lower() or 'use' in question.lower():
            return """To use Magnus features:

1. Run the dashboard: `streamlit run dashboard.py`
2. Navigate using the sidebar menu
3. Use AVA via Telegram bot or direct queries
4. Check documentation files for detailed guides

What specific feature would you like to learn about?"""

        elif 'database' in question.lower() or 'schema' in question.lower():
            return """Magnus uses PostgreSQL with multiple schemas:

- **development_tasks** - Task and QA tracking
- **xtrades** - Professional trader alerts
- **earnings** - Earnings calendar
- **kalshi** - Prediction markets
- **options_data** - Market data cache
- **supply_demand** - Technical analysis zones

Check `src/xtrades_schema.sql` and related schema files for details."""

        else:
            return """I can help you understand Magnus project features and usage.

Try asking:
- "What features does Magnus have?"
- "How do I find CSP opportunities?"
- "What's in the database?"
- "How does the AI agent work?"
- "Where is the code for [feature]?"

Or describe what you're trying to do!"""

    def get_feature_info(self, feature_name: str) -> Dict:
        """Get detailed information about a specific Magnus feature"""
        # Feature lookup could query RAG specifically for that feature
        question = f"Explain the {feature_name} feature in Magnus, including files and capabilities"
        return self.answer_project_question(question)

    def get_usage_guide(self, task: str) -> Dict:
        """Get step-by-step guide for accomplishing a task"""
        question = f"How do I {task} in Magnus? Provide step-by-step instructions."
        return self.answer_project_question(question)

    def search_code(self, search_term: str) -> Dict:
        """Search for code references related to a term"""
        question = f"Where in the Magnus codebase can I find code related to: {search_term}?"
        return self.answer_project_question(question)


def integrate_with_ava(ava_handler):
    """
    Integrate enhanced project handler with existing AVA NLP handler

    Args:
        ava_handler: Existing AVA NaturalLanguageHandler instance

    Returns:
        Enhanced handler that can answer project questions
    """
    project_handler = EnhancedProjectHandler()

    # Store original parse_intent method
    original_parse_intent = ava_handler.parse_intent

    def enhanced_parse_intent(user_text: str, context: Optional[Dict] = None) -> Dict:
        """Enhanced intent parsing that handles project questions"""

        # Check if this is a project-specific question
        if project_handler.is_project_question(user_text):
            logger.info("Detected project question, using enhanced handler")

            answer_result = project_handler.answer_project_question(user_text)

            return {
                'intent': 'PROJECT_QUESTION',
                'confidence': 0.9 if answer_result['success'] else 0.5,
                'entities': {},
                'response_hint': answer_result.get('answer', ''),
                'sources': answer_result.get('sources', []),
                'model_used': answer_result.get('model_used', 'unknown'),
                'cost': answer_result.get('cost', 0.0),
                'original_query': user_text
            }

        # Otherwise use original handler
        return original_parse_intent(user_text, context)

    # Replace parse_intent method
    ava_handler.parse_intent = enhanced_parse_intent
    ava_handler.project_handler = project_handler

    logger.info("[OK] AVA enhanced with project knowledge capabilities")
    return ava_handler


def main():
    """Test enhanced project handler"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("=" * 80)
    print("ENHANCED PROJECT HANDLER TEST")
    print("=" * 80)
    print()

    handler = EnhancedProjectHandler()

    # Test questions
    test_questions = [
        "What features does Magnus have?",
        "How do I find CSP opportunities?",
        "Where is the positions page code?",
        "Explain the database schema",
        "How does AVA work?"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n[Question {i}] {question}")
        print("-" * 80)

        result = handler.answer_project_question(question)

        if result['success']:
            print(result['answer'])
            if result['sources']:
                print(f"\n[Sources: {len(result['sources'])} context sections used]")
            print(f"[Model: {result['model_used']}, Cost: ${result['cost']:.4f}]")
        else:
            print(f"[Error: {result.get('error')}]")
            print(result['answer'])

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
